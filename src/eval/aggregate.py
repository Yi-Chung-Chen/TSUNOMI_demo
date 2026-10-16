"""Write per-date and combined, agent-window-weighted Demo comparisons."""
from collections import defaultdict
import csv
import json
from pathlib import Path

METRICS = ['minADE_m', 'ADE_m', 'FDE_m', 'miss_rate', 'mean_rollout_miss_rate']


def aggregate(records):
    sums = defaultdict(lambda: dict(n_agents=0, totals=defaultdict(float), weights=defaultdict(int)))
    for record in records:
        for date in [record['date'], 'combined']:
            key = (record['model'], date, record['group'], record['horizon_s'], record['axis'])
            entry = sums[key]
            entry['n_agents'] += record['n_agents']
            for name in METRICS:
                if record[name] is not None:
                    entry['totals'][name] += record[name] * record['n_agents']
                    entry['weights'][name] += record['n_agents']
    rows = []
    for key, value in sorted(sums.items()):
        row = dict(zip(['model', 'date', 'group', 'horizon_s', 'axis'], key))
        row['n_agents'] = value['n_agents']
        for name in METRICS:
            row[name] = value['totals'][name] / value['weights'][name] if value['weights'][name] else None
        rows.append(row)
    return rows


def write_summary(rows, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# BC Demo evaluation', '',
             'BC checkpoint: epoch_016.ckpt; no retraining. Val merges existing windows: 2026-10-03 (488) and 2026-10-04 (424). 10-03 was used to select both checkpoint epoch16 and dictionary configuration K2048/chord heading; dictionary centres were built from train only. 10-04 was not used for either choice. Combined results are not an untouched test score.', '',
             '32 joint rollouts, topk_prob K=5, temperature=1.0. ADE/FDE average over rollouts; minADE is the per-agent minimum whole-trajectory ADE. Horizontal/vertical minima are independent. Miss rate = all 32 horizontal endpoints exceed 500 m at 30 s; mean_rollout_miss_rate is also reported. Targets need continuous GT through each horizon. Counts are agent-window instances; aggregation weights agents, not windows.', '',
             '## Does BC beat constant velocity?', '',
             'Lower is better. This compares minADE, ADE and FDE separately; each comparison is paired on the same eligible agents.', '',
             '| Date | Group | Horizon s | Axis | Agents | minADE: BC / CV / BC wins | ADE: BC / CV / BC wins | FDE: BC / CV / BC wins |',
             '|---|---|---|---|---|---|---|---|']
    lookup = {(r['model'], r['date'], r['group'], r['horizon_s'], r['axis']): r for r in rows}
    comparisons = []
    for bc in rows:
        if bc['model'] != 'bc':
            continue
        cv = lookup[('constant_velocity', bc['date'], bc['group'], bc['horizon_s'], bc['axis'])]
        assert bc['n_agents'] == cv['n_agents']
        cells = []
        comparison = {k: bc[k] for k in ['date', 'group', 'horizon_s', 'axis', 'n_agents']}
        for metric in ['minADE_m', 'ADE_m', 'FDE_m']:
            a, b = bc[metric], cv[metric]
            win = None if a is None else a < b
            comparison[metric + '_bc_beats_cv'] = win
            cells.append('NA' if a is None else f'{a:.3f} / {b:.3f} / {"yes" if win else "no (tie)" if a == b else "no"}')
        comparisons.append(comparison)
        lines.append('| ' + ' | '.join(map(str, [bc['date'], bc['group'], bc['horizon_s'], bc['axis'], bc['n_agents'], *cells])) + ' |')
    lines += ['', '## All metrics', '', '| Model | Date | Group | Horizon s | Axis | Agents | minADE m | ADE m | FDE m | Miss rate | Mean rollout miss rate |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        values = [r[k] for k in ['model', 'date', 'group', 'horizon_s', 'axis', 'n_agents']]
        values += ['NA' if r[k] is None else f'{r[k]:.6f}' for k in METRICS]
        lines.append('| ' + ' | '.join(map(str, values)) + ' |')
    (out / 'summary.md').write_text('\n'.join(lines) + '\n')
    (out / 'bc_vs_cv.json').write_text(json.dumps(comparisons, indent=2) + '\n')
