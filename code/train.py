"""Default recipe: 1,200 steps x 32 sequences x 256 targets = 9,830,400 tokens."""
import argparse
import copy
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score


def update_ema(ema_model, model, decay):
    """Apply a single EMA update using the current online parameters."""
    if not 0.0 <= decay < 1.0:
        raise ValueError('EMA decay must be in [0, 1).')
    with torch.no_grad():
        for name, value in model.state_dict().items():
            ema_value = ema_model.state_dict()[name]
            if value.is_floating_point():
                ema_value.mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                ema_value.copy_(value.detach())


def update_swa(average_model, model, count):
    if count < 1:
        raise ValueError('SWA update count must be positive.')
    with torch.no_grad():
        for name, value in model.state_dict().items():
            average_value = average_model.state_dict()[name]
            if value.is_floating_point():
                average_value.add_((value.detach() - average_value) / count)
            else:
                average_value.copy_(value.detach())


def copy_model_state(target, source):
    with torch.no_grad():
        for target_value, source_value in zip(target.state_dict().values(), source.state_dict().values()):
            target_value.copy_(source_value.detach())


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
    p.add_argument('--dropout', type=float, default=None,
                   help='Dropout probability; 0 disables dropout (overrides config).')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--ema-decay', type=float, default=0.999,
                   help='EMA decay; only used when --averaging-method=ema.')
    p.add_argument('--averaging-method', choices=['none', 'ema', 'swa'], default='none',
                   help='Optional weight averaging method.')
    p.add_argument('--average-start-step', type=int, default=1,
                   help='Step at which to start weight averaging.')
    p.add_argument('--average-every', type=int, default=1,
                   help='Update averaged weights every N optimizer steps.')
    p.add_argument('--eval-every', type=int, default=0,
                   help='Optional validation-curve interval; 0 evaluates only after training.')
    p.add_argument('--checkpoint-every', type=int, default=0,
                   help='Save periodic model checkpoints under run-dir/checkpoints; 0 disables.')
    args = p.parse_args()
    schedule_steps = args.schedule_steps or args.steps
    if (args.steps < 1 or args.batch_size < 1 or schedule_steps < 1
            or args.learning_rate <= 0 or args.weight_decay < 0
            or (args.dropout is not None and not 0.0 <= args.dropout < 1.0)
            or args.checkpoint_every < 0 or args.average_start_step < 1
            or args.average_start_step > args.steps or args.average_every < 1):
        p.error('Steps and batch size must be positive; learning rate must be positive and weight decay non-negative.')
    if args.steps > schedule_steps:
        p.error('--steps cannot exceed --schedule-steps.')
    if args.averaging_method == 'ema' and not 0.0 < args.ema_decay < 1.0:
        p.error('--ema-decay must be in (0, 1) when --averaging-method=ema.')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Run directory already contains results. Use a new --run-dir.')
    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text())
    config['dropout'] = config.get('dropout', 0.0) if args.dropout is None else args.dropout
    model, implementation_sha = make_model(args.implementation, config, device)
    averaged_model = None
    averaging_started = False
    average_count = 0
    averaging_enabled = args.averaging_method != 'none'
    if averaging_enabled:
        averaged_model = copy.deepcopy(model).to(device)
        averaged_model.eval()
        averaged_model.requires_grad_(False)
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
        if averaged_model is not None and completed_step >= args.average_start_step:
            if not averaging_started:
                copy_model_state(averaged_model, model)
                averaging_started = True
                average_count = 1
            elif completed_step % args.average_every == 0 or completed_step == args.steps:
                if args.averaging_method == 'ema':
                    update_ema(averaged_model, model, args.ema_decay)
                else:
                    average_count += 1
                    update_swa(averaged_model, model, average_count)
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
    averaged_checkpoint = None
    averaged_validation = None
    if averaged_model is not None:
        averaged_validation = score(averaged_model,*data['validation'],device,'fp32')
        averaged_validation.pop('window_nll_nats')
        averaged_model = averaged_model.cpu()
        averaged_checkpoint = args.run_dir/f'{args.averaging_method}-checkpoint.pt'
        torch.save({'protocol':PROTOCOL,'implementation':args.implementation,'config':config,
                    'model':averaged_model.state_dict(),'seed':args.seed,
                    'averaging_method':args.averaging_method,
                    'averaging_decay':args.ema_decay if args.averaging_method == 'ema' else None,
                    'averaging_start_step':args.average_start_step,
                    'averaging_every':args.average_every,
                    'train_tokens':args.steps*args.batch_size*256},averaged_checkpoint)
    result = {'protocol':PROTOCOL,'implementation':args.implementation,'config':config,'seed':args.seed,
              'parameters':sum(p.numel() for p in model.parameters()),'precision':precision,
              'steps':args.steps,'schedule_steps':schedule_steps,'batch_size':args.batch_size,
              'learning_rate':args.learning_rate,'weight_decay':args.weight_decay,
              'averaging_decay':args.ema_decay if averaging_enabled and args.averaging_method == 'ema' else None,
              'averaging_method':args.averaging_method,
              'averaging_start_step':args.average_start_step if averaging_enabled else None,
              'averaging_every':args.average_every if averaging_enabled else None,
              'train_tokens':args.steps*args.batch_size*256,'preparation_seconds':preparation_seconds,
              'train_seconds':train_seconds,'validation':validation,'history':history,
              'validation_history':validation_history,
              'intermediate_validation_seconds':intermediate_validation_seconds,
              'process_seconds':time.perf_counter()-total_started,
              'torch_version':str(torch.__version__),'threads':args.threads,
              'checkpoint_sha256':sha(checkpoint),'implementation_sha256':implementation_sha,
              **device_metrics(device)}
    if averaged_checkpoint is not None:
        result['averaged_checkpoint'] = averaged_checkpoint.name
        result['averaged_validation'] = averaged_validation
        result['averaged_checkpoint_sha256'] = sha(averaged_checkpoint)
    (args.run_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result|{'history':[]},indent=2),flush=True)


if __name__ == '__main__':
    main()