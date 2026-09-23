"""Default recipe: 1,200 steps x 32 sequences x 256 targets = 9,830,400 tokens."""
import argparse
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score


def main():
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT/'configs/baseline.json')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/baseline-s17')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto','fp32','bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--schedule-steps', type=int, default=None,
                   help='Cosine decay horizon; defaults to the number of updates.')
    p.add_argument('--learning-rate', type=float, default=0.001,
                   help='Peak AdamW learning rate before cosine decay.')
    p.add_argument('--weight-decay', type=float, default=0.1,
                   help='AdamW decoupled weight decay.')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--eval-every', type=int, default=0,
                   help='Optional validation-curve interval; 0 evaluates only after training.')
    p.add_argument('--checkpoint-every', type=int, default=0,
                   help='Save periodic model checkpoints under run-dir/checkpoints; 0 disables.')
    args = p.parse_args()
    schedule_steps = args.schedule_steps or args.steps
    if (args.steps < 1 or args.batch_size < 1 or schedule_steps < 1
            or args.learning_rate <= 0 or args.weight_decay < 0
            or args.checkpoint_every < 0):
        p.error('Steps and batch size must be positive; learning rate must be positive and weight decay non-negative.')
    if args.steps > schedule_steps:
        p.error('--steps cannot exceed --schedule-steps.')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Run directory already contains results. Use a new --run-dir.')
    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text())
    model, implementation_sha = make_model(args.implementation, config, device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.run_dir/'checkpoints'
    if args.checkpoint_every:
        checkpoint_dir.mkdir()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter()-prepared
    started = time.perf_counter()
    history = []
    validation_history = []
    intermediate_validation_seconds = 0.
    for step in range(args.steps):
        starts = torch.randint(len(tokens)-257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:,None]+torch.arange(257,device=device)]
        learning_rate = args.learning_rate * min(1.,(step+1)/100) * (.1+.9*.5*(1+math.cos(math.pi*step/schedule_steps)))
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            loss = F.cross_entropy(model(batch[:,:-1]).flatten(0,1).float(),batch[:,1:].flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        completed_step = step + 1
        if args.checkpoint_every and (completed_step % args.checkpoint_every == 0
                          or completed_step == args.steps):
            periodic_state = {name: value.detach().cpu().clone()
                      for name, value in model.state_dict().items()}
            torch.save({'protocol':PROTOCOL,'implementation':args.implementation,
                'config':config,'seed':args.seed,'step':completed_step,
                'model':periodic_state},
                   checkpoint_dir/f'step-{completed_step:06d}.pt')
        if (step+1)%100 == 0 or step+1 == args.steps:
            row = {'step':step+1,'loss':loss.item(),'seconds':time.perf_counter()-started-intermediate_validation_seconds}
            history.append(row)
            print(json.dumps(row),flush=True)
        if args.eval_every > 0 and (step+1)%args.eval_every == 0:
            intermediate = score(model,*data['validation'],device,'fp32')
            intermediate.pop('window_nll_nats')
            intermediate_validation_seconds += intermediate['seconds']
            validation_history.append({'step':step+1,**intermediate})
            print(json.dumps({'validation':validation_history[-1]}),flush=True)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter()-started-intermediate_validation_seconds
    validation = score(model,*data['validation'],device,'fp32')
    validation.pop('window_nll_nats')
    checkpoint = args.run_dir/'checkpoint.pt'
    torch.save({'protocol':PROTOCOL,'implementation':args.implementation,'config':config,
                'model':model.cpu().state_dict(),'seed':args.seed,
                'train_tokens':args.steps*args.batch_size*256},checkpoint)
    result = {'protocol':PROTOCOL,'implementation':args.implementation,'config':config,'seed':args.seed,
              'parameters':sum(p.numel() for p in model.parameters()),'precision':precision,
              'steps':args.steps,'schedule_steps':schedule_steps,'batch_size':args.batch_size,
              'learning_rate':args.learning_rate,'weight_decay':args.weight_decay,
              'train_tokens':args.steps*args.batch_size*256,'preparation_seconds':preparation_seconds,
              'train_seconds':train_seconds,'validation':validation,'history':history,
              'validation_history':validation_history,
              'intermediate_validation_seconds':intermediate_validation_seconds,
              'process_seconds':time.perf_counter()-total_started,
              'torch_version':str(torch.__version__),'threads':args.threads,
              'checkpoint_sha256':sha(checkpoint),'implementation_sha256':implementation_sha,
              **device_metrics(device)}
    (args.run_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result|{'history':[]},indent=2),flush=True)


if __name__ == '__main__':
    main()
