"""One command: existing-index merge, BC rollout, CV/GT baselines, metrics, plots."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from scripts.merge_demo_val import merge, protected_hashes
from src.data.dataset import AirDataset
from src.model.air_smart import AirSMART
from src.rollout.closed_loop import rollout
from src.rollout.baselines import baselines
from src.rollout.rollout_io import save_rollout_window
from src.eval.load_rollout import load_rollout
from src.eval.metrics.horizon_ade import compute
from src.eval.aggregate import aggregate, write_summary
from src.eval.demo_plots import write_plots


def window_seed(seed, scenario, start):
    return int.from_bytes(hashlib.sha256(f'{seed}/{scenario}/{start}'.encode()).digest()[:8], 'little') % (2**63 - 1)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', type=Path, default=Path('outputs/air_pre_bc/checkpoints/epoch_016.ckpt'))
    p.add_argument('--seed', type=int, default=817)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--max-windows', type=int, default=0, help='0 means full912; smoke outputs get separate directories')
    p.add_argument('--output-root', type=Path, default=Path('outputs'))
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    receipt = merge()
    before = protected_hashes()
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    assert ckpt['epoch'] == 16
    model = AirSMART(**ckpt['hyper_parameters'])
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval().requires_grad_(False)
    dataset = AirDataset('data/processed', 'data/splits', 'val')
    assert len(dataset) == 912
    root = args.output_root
    if args.max_windows:
        root = root / 'demo_smoke'
    records, bc_paths, dates = [], [], {}
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    started = time.monotonic()
    limit = min(args.max_windows, len(dataset)) if args.max_windows else len(dataset)
    for index in range(limit):
        graph = dataset[index]
        scenario, start, date = graph['scenario'], graph['start_frame'], graph['date']
        name = f'{scenario}_f{start:04d}.npz'
        seed = window_seed(args.seed, scenario, start)
        bc_path = root / 'demo_rollouts/bc' / name
        reuse = False
        if bc_path.exists():
            cached = load_rollout(bc_path)
            reuse = cached['meta']['seed'] == seed and cached['meta']['checkpoint_sha256'] == checkpoint_hash
        if not reuse:
            prediction = rollout(model, graph, n_rollouts=32, num_k=5, temp=1., seed=seed)
            save_rollout_window(graph, prediction, bc_path, 'bc', seed, checkpoint_hash)
        bc_paths.append(bc_path)
        output_paths = dict(bc=bc_path)
        for baseline, prediction in baselines(graph).items():
            path = root / 'demo_rollouts' / baseline / name
            save_rollout_window(graph, prediction, path, baseline, seed)
            output_paths[baseline] = path
        for kind, path in output_paths.items():
            rows = compute(load_rollout(path))
            if kind == 'gt_replay':
                assert all(r[k] in (0., None) for r in rows for k in ['minADE_m', 'ADE_m', 'FDE_m', 'miss_rate', 'mean_rollout_miss_rate'])
            records.extend(dict(model=kind, **r) for r in rows)
        dates[date] = dates.get(date, 0) + 1
        if (index + 1) % 25 == 0 or index + 1 == limit:
            print(json.dumps(dict(windows=index + 1, total=limit, seconds=round(time.monotonic() - started, 1))), flush=True)
    summary = aggregate(records)
    write_summary(summary, root / 'eval')
    selection = write_plots(bc_paths, root / 'demo')
    assert protected_hashes() == before == receipt['protected_sha256']
    acceptance = dict(windows=limit, dates=dates, rollouts=32, num_k=5, temp=1., device='cpu',
                      checkpoint=str(args.checkpoint), checkpoint_epoch=16, plot_count=len(selection),
                      elapsed_s=time.monotonic() - started, seed=args.seed, plot_seed=0, protected_sha256=before,
                      gt_replay_exact_zero=True, no_training=True, full_demo=limit == 912)
    (root / 'eval/acceptance.json').write_text(json.dumps(acceptance, indent=2) + '\n')
    if limit == 912:
        assert dates == {'2026-10-03': 488, '2026-10-04': 424}
        assert len(selection) == 6
        shutil.copy2(root / 'eval/summary.md', 'docs/demo_results.md')
        Path('docs/demo_acceptance.json').write_text(json.dumps(acceptance, indent=2) + '\n')
    print(json.dumps(acceptance, indent=2), flush=True)


if __name__ == '__main__':
    main()
