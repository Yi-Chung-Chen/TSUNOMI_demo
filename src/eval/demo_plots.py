"""Seeded GT-phase stratified window selection; never rank prediction errors."""
import json
from pathlib import Path
import numpy as np
from src.eval.load_rollout import load_rollout


def choose_windows(paths, seed=0):
    candidates = []
    for path in sorted(paths):
        d = load_rollout(path)
        for agent in np.flatnonzero(d['target_mask']):
            delta = float(d['gt_position'][agent, 48, 2] - d['gt_position'][agent, 18, 2])
            kind = 'ground' if d['on_ground'][agent, 18] else 'climbing' if delta > 20. else 'descending' if delta < -20. else 'level'
            candidates.append(dict(path=str(path), agent=int(agent), agent_id=str(d['agent_id'][agent]), kind=kind,
                                   date=str(d['date']), scenario=str(d['scenario']), start_frame=int(d['start_frame'])))
    rng, selected, used = np.random.default_rng(seed), [], set()
    for kind in ['climbing', 'descending', 'level', 'ground']:
        pool = [r for r in candidates if r['kind'] == kind and r['path'] not in used]
        if pool:
            row = pool[int(rng.integers(len(pool)))]
            selected.append(row)
            used.add(row['path'])
    while len(selected) < min(6, len(paths)):
        pool = [r for r in candidates if r['path'] not in used]
        row = pool[int(rng.integers(len(pool)))]
        selected.append(row)
        used.add(row['path'])
    return selected


def write_plots(paths, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    selected = choose_windows(paths)
    (out / 'selection.json').write_text(json.dumps(dict(seed=0, rule='ground at current; otherwise dz10s above20m/below-20m/level; no error ranking', windows=selected), indent=2) + '\n')
    future_t, history_t = np.arange(1, 91) / 3., np.arange(-18, 1) / 3.
    for row in selected:
        d = load_rollout(row['path'])
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        chosen = row['agent']
        for agent in range(len(d['agent_id'])):
            highlight = agent == chosen
            gt = d['gt_position'][agent].astype(np.float64).copy()
            gt[~d['valid_mask'][agent]] = np.nan
            axes[0].plot(gt[:19, 0], gt[:19, 1], color='black' if highlight else '.45', lw=2 if highlight else .8,
                         label='Selected history' if highlight else None)
            axes[0].plot(gt[18:, 0], gt[18:, 1], color='#d97706' if highlight else '#b9a183', lw=2 if highlight else .8,
                         label='Selected GT future' if highlight else None)
            axes[1].plot(history_t, gt[:19, 2], color='black' if highlight else '.45', lw=2 if highlight else .8)
            axes[1].plot(np.r_[0., future_t], gt[18:, 2], color='#d97706' if highlight else '#b9a183', lw=2 if highlight else .8)
            for r, pred in enumerate(d['position'][:, agent]):
                axes[0].plot(pred[:, 0], pred[:, 1], color='#2563eb' if highlight else '.6', alpha=.22 if highlight else .07,
                             lw=.8 if highlight else .5, label='Selected BC (32 rollouts)' if highlight and r == 0 else None)
                axes[1].plot(future_t, pred[:, 2], color='#2563eb' if highlight else '.6', alpha=.22 if highlight else .07,
                             lw=.8 if highlight else .5)
            axes[0].scatter(gt[18, 0], gt[18, 1], s=45 if highlight else 10, color='black', marker='*' if highlight else 'o')
        axes[0].set(xlabel='East (m)', ylabel='North (m)', title='ENU trajectories', aspect='equal')
        axes[0].legend(fontsize=8)
        axes[1].set(xlabel='Time from current (s)', ylabel='Height above reference ground (m)', title='Altitude vs time')
        axes[1].axvline(0, color='.4', ls=':', lw=.8)
        axes[1].axhline(0, color='.7', ls=':', lw=.8)
        for ax in axes:
            ax.grid(alpha=.15)
        fig.suptitle(f'{row["date"]} / {row["scenario"]} / start {row["start_frame"]} / {row["kind"]} / target {row["agent_id"]}')
        fig.tight_layout()
        fig.savefig(out / f'{row["kind"]}_{row["scenario"]}_f{row["start_frame"]:04d}.png', dpi=150)
        plt.close(fig)
    return selected
