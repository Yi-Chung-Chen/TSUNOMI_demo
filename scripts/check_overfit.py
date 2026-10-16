"""M5 single-batch unsmoothed overfit acceptance (CPU or GPU)."""
import argparse
import json
from pathlib import Path
import torch
from torch_geometric.loader import DataLoader
from src.data.dataset import AirDataset
from src.model.air_smart import AirSMART
from src.metrics.cross_entropy import token_cross_entropy


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--device', default='cpu')
    p.add_argument('--max-steps', type=int, default=1200)
    p.add_argument('--output-dir', type=Path, default=Path('outputs/m5_overfit'))
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(817)
    model = AirSMART('assets/motion_dict_K2048_s6.pkl', dropout=0., label_smoothing=0.).to(args.device)
    dataset = AirDataset('data/processed', 'data/splits', 'train')
    batch = next(iter(DataLoader([dataset[i] for i in range(4)], batch_size=4))).to(args.device)
    tokens = model.token_processor.tokenize_agent(batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=0.)
    history = []
    for step in range(args.max_steps):
        optimizer.zero_grad()
        logits = model.decoder(tokens, model.token_features)
        result = token_cross_entropy(logits, tokens, model.current_node, 0.)
        result['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step % 25 == 0 or (result['unsmoothed_ce'] < .03 and result['accuracy'] >= .99):
            row = dict(step=step, ce=float(result['unsmoothed_ce']), accuracy=float(result['accuracy']), tokens=result['count'])
            print(json.dumps(row), flush=True)
            history.append(row)
        if result['unsmoothed_ce'] < .03 and result['accuracy'] >= .99:
            break
    else:
        raise RuntimeError('Overfit acceptance did not converge')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'result.json').write_text(json.dumps(dict(device=args.device, history=history), indent=2) + '\n')
    print('OVERFIT_ACCEPTANCE_PASSED', flush=True)


if __name__ == '__main__':
    main()
