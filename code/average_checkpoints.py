"""Average late training checkpoints into one deployable checkpoint."""
import argparse
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    payloads = [torch.load(path, map_location='cpu', weights_only=False)
                for path in args.checkpoints]
    reference = payloads[0]
    reference_keys = tuple(reference['model'])
    if not reference_keys:
        raise ValueError('Checkpoint contains no model parameters.')
    for payload in payloads[1:]:
        if tuple(payload['model']) != reference_keys:
            raise ValueError('All checkpoints must have identical model keys.')

    averaged = {}
    for name in reference_keys:
        tensors = [payload['model'][name] for payload in payloads]
        if not all(tensor.shape == tensors[0].shape for tensor in tensors):
            raise ValueError(f'Parameter shape mismatch for {name}.')
        if tensors[0].is_floating_point():
            total = torch.zeros_like(tensors[0], dtype=torch.float64)
            for tensor in tensors:
                total.add_(tensor.to(dtype=torch.float64))
            averaged[name] = (total / len(tensors)).to(dtype=tensors[0].dtype)
        else:
            if not all(torch.equal(tensor, tensors[0]) for tensor in tensors[1:]):
                raise ValueError(f'Non-floating buffer differs for {name}.')
            averaged[name] = tensors[0].clone()

    result = dict(reference)
    result['model'] = averaged
    result['averaged_checkpoints'] = [str(path) for path in args.checkpoints]
    result['average_count'] = len(payloads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(f'averaged {len(payloads)} checkpoints -> {args.output}')


if __name__ == '__main__':
    main()