"""Train-only 3D K-disk vocabulary builder, adapted from PRIV motion builder."""
import argparse
import json
import pickle
from pathlib import Path
import subprocess
import numpy as np
from src.common.geometry import contours, rotate, wrap_angle, causal_headings
from src.common.units import to_internal, FRAME_HZ


def extract_pool(processed_dir, manifest, shift, pos_scale=10., max_vertical_speed=40.,
                 max_horizontal_speed=200., max_segments=500000, seed=42):
    positions, data_heads, chord_heads = [], [], []
    stats = dict(candidate_segments=0, invalid_segments=0, vertical_outliers=0, horizontal_outliers=0)
    for name in Path(manifest).read_text().splitlines():
        with np.load(Path(processed_dir) / f'{name}.npz', allow_pickle=False) as d:
            p, v, h = d['pos'], d['valid'], d['heading']
            ch = causal_headings(p, h, v, d['on_ground'], shift)
            for start in range(0, p.shape[1] - shift, shift):
                seg = p[:, start:start + shift + 1]
                ok = v[:, start:start + shift + 1].all(-1)
                stats['candidate_segments'] += len(v)
                stats['invalid_segments'] += int((~ok).sum())
                vel = np.diff(seg, axis=1) * FRAME_HZ
                bad_z = np.abs(vel[..., 2]).max(-1) > max_vertical_speed
                bad_xy = np.linalg.norm(vel[..., :2], axis=-1).max(-1) > max_horizontal_speed
                stats['vertical_outliers'] += int((ok & bad_z).sum())
                stats['horizontal_outliers'] += int((ok & ~bad_z & bad_xy).sum())
                ok &= ~bad_z & ~bad_xy
                if ok.any():
                    positions.append(to_internal(seg[ok], pos_scale))
                    data_heads.append(h[ok, start:start + shift + 1])
                    chord_heads.append(ch[ok, start:start + shift + 1])
    p, h, ch = map(np.concatenate, [positions, data_heads, chord_heads])
    stats['n_segments_total'] = len(p)
    if len(p) > max_segments:
        ids = np.random.default_rng(seed).choice(len(p), max_segments, replace=False)
        p, h, ch = p[ids], h[ids], ch[ids]
    stats['n_segments_used'] = len(p)
    return p, h, ch, stats


def normalize_segments(position, heading):
    xy = rotate(position[..., :2] - position[:, :1, :2], -heading[:, :1])
    traj = np.concatenate([xy, wrap_angle(heading - heading[:, :1])[..., None]], -1)
    dz = position[..., 2] - position[:, :1, 2]
    return traj.astype(np.float32), dz.astype(np.float32)


def kdisk_cluster(traj, dz, K, tol=.5, radius=2., lambda_z=1., seed=42):
    """Deterministic K-disk; token zero stays exactly stationary.

    Each centre is the arithmetic xy/dz and circular heading mean of consumed
    samples. Subsequent prefixes are valid dictionaries with identical sampling.
    """
    endpoints = contours(traj[:, -1], radius)
    out_traj = [np.zeros_like(traj[0])]
    out_dz = [np.zeros_like(dz[0])]
    zero = contours(out_traj[0][-1], radius)
    dist = np.linalg.norm(endpoints - zero, axis=-1).mean(-1) + lambda_z * np.abs(dz[:, -1])
    remaining = np.flatnonzero(dist > tol)
    fractions = [len(remaining) / len(traj)]
    rng = np.random.default_rng(seed)
    for i in range(1, K):
        if not len(remaining):
            return None
        pick = remaining[rng.integers(len(remaining))]
        diff = endpoints[remaining] - endpoints[pick]
        distance = np.sqrt((diff * diff).sum(-1)).mean(-1)
        distance += lambda_z * np.abs(dz[remaining, -1] - dz[pick, -1])
        inside = remaining[distance <= tol]
        centre = traj[inside].mean(0)
        centre[:, 2] = np.arctan2(np.sin(traj[inside, :, 2]).mean(0), np.cos(traj[inside, :, 2]).mean(0))
        centre[0] = 0
        z = dz[inside].mean(0)
        z[0] = 0
        out_traj.append(centre)
        out_dz.append(z)
        remaining = remaining[distance > tol]
        fractions.append(len(remaining) / len(traj))
        if (i + 1) % 512 == 0:
            print(f'  K={i + 1}, remaining={len(remaining)}, tol={tol:.5f}', flush=True)
    return np.stack(out_traj), np.stack(out_dz), fractions


def build_grid(processed_dir, split_dir, output_dir, shifts=(3, 6, 9), ks=(512, 1024, 2048),
               tol=.5, radius=2., lambda_z=1., pos_scale=10., seed=42,
               max_segments=500000, max_vertical_speed=40., max_horizontal_speed=200.):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    paths = []
    for shift in shifts:
        p, h, ch, stats = extract_pool(processed_dir, Path(split_dir) / 'train_manifest.txt', shift,
                                     pos_scale, max_vertical_speed, max_horizontal_speed, max_segments, seed)
        print(f'shift={shift}: {stats}', flush=True)
        for source, headings in [('data', h), ('chord', ch)]:
            traj, dz = normalize_segments(p, headings)
            actual_tol = tol
            for attempt in range(40):
                result = kdisk_cluster(traj, dz, max(ks), actual_tol, radius, lambda_z, seed)
                if result is not None:
                    break
                actual_tol *= .8
            else:
                raise RuntimeError('Could not obtain exact K after tolerance retries')
            centres, z, fractions = result
            for k in ks:
                meta = dict(shift=shift, pos_scale=pos_scale, radius=radius, lambda_z=lambda_z, K=k,
                            git_hash=git_hash, heading_source=source, tol=actual_tol, seed=seed,
                            max_vertical_speed=max_vertical_speed, max_horizontal_speed=max_horizontal_speed,
                            unclustered_fraction=fractions[k - 1], **stats)
                dictionary = dict(cluster_centers=contours(centres[:k], radius).astype(np.float32),
                                  traj=centres[:k].copy(), dz=z[:k].copy(), meta=meta)
                path = out / f'motion_dict_K{k}_s{shift}_{source}.pkl'
                with path.open('wb') as stream:
                    pickle.dump(dictionary, stream)
                paths.append(str(path))
                print(f'Saved {path}', flush=True)
    return paths


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--processed-dir', default='data/processed')
    p.add_argument('--split-dir', default='data/splits')
    p.add_argument('--output-dir', default='data/motion_dict')
    p.add_argument('--shifts', type=int, nargs='+', default=[3, 6, 9])
    p.add_argument('--ks', type=int, nargs='+', default=[512, 1024, 2048])
    p.add_argument('--tol', type=float, default=.5)
    p.add_argument('--radius', type=float, default=2.)
    p.add_argument('--lambda-z', type=float, default=1.)
    p.add_argument('--pos-scale', type=float, default=10.)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-segments', type=int, default=500000)
    p.add_argument('--max-vertical-speed', type=float, default=40.)
    p.add_argument('--max-horizontal-speed', type=float, default=200.)
    args = p.parse_args()
    paths = build_grid(**vars(args))
    print(json.dumps(paths, indent=2))


if __name__ == '__main__':
    main()
