"""Draft-v1 top-down animations from saved rollouts; never invokes a model."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MarkerPath
from matplotlib.transforms import Affine2D
import numpy as np
from PIL import Image, ImageEnhance
import requests

from src.eval.load_rollout import load_rollout

ORIGIN = (41.980103, -87.903868)
RADIUS = 6378137.
SIZE = (1200, 900)
MAP_RECT = (.025, .055, .95, .86)
PALETTE = ['#007f86', '#b74b34', '#6652a3', '#278048', '#ab3581', '#ad7900', '#2368ac', '#764b36']
# East-pointing aircraft silhouette, heading is counter-clockwise from east.
PLANE = MarkerPath(np.array([(1, 0), (.65, .12), (.12, .12), (-.35, .85),
    (-.55, .85), (-.35, .12), (-.75, .12), (-.95, .35), (-1, .35),
    (-.9, 0), (-1, -.35), (-.95, -.35), (-.75, -.12), (-.35, -.12),
    (-.55, -.85), (-.35, -.85), (.12, -.12), (.65, -.12), (1, 0)]))


def enu_to_pixel(xy, zoom):
    xy = np.asarray(xy, dtype=float)
    lat = ORIGIN[0] + np.degrees(xy[..., 1] / RADIUS)
    lon = ORIGIN[1] + np.degrees(xy[..., 0] / (RADIUS * np.cos(np.radians(ORIGIN[0]))))
    scale = 256 * 2**zoom
    return np.stack(((lon + 180) / 360 * scale,
        (1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * scale), axis=-1)


def pixel_to_enu(pixels, zoom):
    pixels = np.asarray(pixels, dtype=float)
    scale = 256 * 2**zoom
    lon = pixels[..., 0] / scale * 360 - 180
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * pixels[..., 1] / scale))))
    return np.stack((np.radians(lon - ORIGIN[1]) * RADIUS * np.cos(np.radians(ORIGIN[0])),
        np.radians(lat - ORIGIN[0]) * RADIUS), axis=-1)


def choose_rollout(data):
    mask = data['valid_mask'][:, 19:] & data['target_mask'][:, None]
    counts = mask.sum(axis=1)
    if not (counts > 0).any():
        raise ValueError('No valid target future')
    errors = np.linalg.norm(data['position'] - data['gt_position'][None, :, 19:], axis=-1)
    # Equal weight per target aircraft, using each aircraft's full valid 3D horizon.
    scores = ((errors * mask).sum(axis=2)[:, counts > 0] / counts[counts > 0]).mean(axis=1)
    return int(scores.argmin()), scores


def frame_indices(skip=1):
    if skip not in (1, 2):
        raise ValueError('skip must be 1 or 2')
    return list(range(0, 109, skip))


def view_bounds(points, aspect=SIZE[0]*MAP_RECT[2]/(SIZE[1]*MAP_RECT[3])):
    low, high = np.min(points, axis=0), np.max(points, axis=0)
    center = (low + high)/2
    span = np.maximum((high-low)*1.2, 20.)
    span = np.array([max(span[0], span[1]*aspect), max(span[1], span[0]/aspect)])
    return np.stack((center-span/2, center+span/2))


def basemap(bounds, cache):
    span = bounds[1]-bounds[0]
    world_m = 2*np.pi*RADIUS*np.cos(np.radians(ORIGIN[0]))
    zoom = max(0, min(18, int(math.floor(math.log2(min(1140/span[0], 774/span[1])*world_m/256)))))
    p = enu_to_pixel(bounds, zoom)
    left, right = sorted(p[:, 0]); top, bottom = sorted(p[:, 1])
    x0, x1, y0, y1 = int(left//256), int(right//256), int(top//256), int(bottom//256)
    mosaic = Image.new('RGB', ((x1-x0+1)*256, (y1-y0+1)*256))
    with requests.Session() as session:
        session.headers['User-Agent'] = 'sygard-motion-demo/1.0 (research visualization; cached, 10 scenes)'
        for x in range(x0, x1+1):
            for y in range(y0, y1+1):
                target = cache / str(zoom) / str(x) / f'{y}.png'
                if not target.exists():
                    response = session.get(f'https://tile.openstreetmap.org/{zoom}/{x}/{y}.png', timeout=30)
                    response.raise_for_status()
                    tile = Image.open(io.BytesIO(response.content)).convert('RGB')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    tile.save(target)
                with Image.open(target) as tile:
                    mosaic.paste(tile.convert('RGB'), ((x-x0)*256, (y-y0)*256))
    # Crop to exact projected bounds, avoiding stretching entire boundary tiles.
    crop = mosaic.crop((left-x0*256, top-y0*256, right-x0*256, bottom-y0*256))
    faded = Image.blend(ImageEnhance.Color(crop).enhance(.25), Image.new('RGB', crop.size, 'white'), .60)
    return faded, zoom, np.array([[left, top], [right, bottom]])


def select_windows(root):
    selected = [root/r['path'] for r in json.loads((root/'outputs/demo/selection.json').read_text())['windows']]
    pool = []
    for path in sorted((root/'outputs/demo_rollouts/bc').glob('*.npz')):
        if path in selected:
            continue
        with np.load(path, allow_pickle=False) as data:
            if np.count_nonzero(data['valid_mask'].any(axis=1)) >= 3:
                pool.append(path)
    selected += [pool[i] for i in np.random.default_rng(0).choice(len(pool), 4, replace=False)]
    assert len(selected) == len(set(selected)) == 10
    return selected


def render_window(data, cache, skip=1):
    rollout, scores = choose_rollout(data)
    gt = data['gt_position'][..., :2].astype(float).copy()
    valid = data['valid_mask']
    gt[~valid] = np.nan
    pred = data['position'][rollout, ..., :2].astype(float).copy()
    # Predictions stop at the first missing GT after the current time.
    pred_valid = np.logical_and.accumulate(valid[:, 18:], axis=1)[:, 1:]
    pred[~pred_valid] = np.nan
    points = np.concatenate((gt[valid], data['position'][rollout, valid.any(axis=1), :, :2].reshape(-1, 2)))
    bounds = view_bounds(points)
    image, zoom, pb = basemap(bounds, cache)
    gtpx, predpx = enu_to_pixel(gt, zoom), enu_to_pixel(pred, zoom)
    fig = plt.figure(figsize=(12, 9), dpi=100, facecolor='#f5f8f7')
    ax = fig.add_axes(MAP_RECT)
    ax.imshow(image, extent=(pb[0,0], pb[1,0], pb[1,1], pb[0,1]))
    ax.set(xlim=(pb[0,0], pb[1,0]), ylim=(pb[1,1], pb[0,1]), aspect='equal')
    ax.axis('off')
    fig.text(.035, .955, f'{data["date"]} · {data["scenario"]}', fontsize=17, color='#233e48', weight='medium')
    fig.text(.035, .025, '━━  ground truth      ┄┄  prediction', fontsize=10, color='#526873')
    fig.text(.975, .025, '© OpenStreetMap contributors', ha='right', fontsize=9, color='#526873')
    clock = fig.text(.965, .955, '', ha='right', fontsize=14, color='#233e48', animated=True)
    artists = [clock]
    tracks = []
    for a, agent in enumerate(data['agent_id']):
        if not valid[a].any():
            continue
        index = int(hashlib.sha256(str(agent).encode()).hexdigest()[:8], 16) % len(PALETTE)
        color = np.array(matplotlib.colors.to_rgb(PALETTE[index])); tint = color*.55+.45
        ax.plot(*gtpx[a].T, color=color, alpha=.15, lw=1.2)
        truth, = ax.plot([], [], color=color, lw=1.8, animated=True)
        future, = ax.plot([], [], color=tint, lw=1.45, ls=(0,(4,3)), animated=True)
        icon, = ax.plot([], [], marker=PLANE, ms=17, mfc=color, mec='white', mew=.6, ls='', animated=True)
        ghost, = ax.plot([], [], marker=PLANE, ms=16, mfc='white', mec=tint, mew=1.3, ls='', animated=True)
        label = ax.text(0, 0, str(agent), fontsize=8, color=color, animated=True,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=.65, pad=1))
        tracks.append((a, truth, future, icon, ghost, label))
        artists.extend([truth, future, ghost, icon, label])
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(fig.bbox)
    frames = []
    for t in frame_indices(skip):
        fig.canvas.restore_region(background)
        clock.set_text(f't = {(t-18)/3:+.1f} s')
        for a, truth, future, icon, ghost, label in tracks:
            truth.set_data(*gtpx[a, :t+1].T)
            k = max(0, t-18)
            trail = np.concatenate((gtpx[a, 18:19], predpx[a, :k])) if k else np.empty((0, 2))
            future.set_data(*trail.T)
            icon.set_visible(bool(valid[a,t])); label.set_visible(bool(valid[a,t]))
            if valid[a,t]:
                icon.set_data([gtpx[a,t,0]], [gtpx[a,t,1]])
                icon.set_marker(PLANE.transformed(Affine2D().rotate(float(data['gt_heading'][a,t]))))
                label.set_position(gtpx[a,t]+np.array([12, -12]))
            show_pred = k > 0 and pred_valid[a,k-1]
            ghost.set_visible(bool(show_pred))
            if show_pred:
                ghost.set_data([predpx[a,k-1,0]], [predpx[a,k-1,1]])
                ghost.set_marker(PLANE.transformed(Affine2D().rotate(float(data['heading'][rollout,a,k-1]))))
        for artist in artists:
            (artist.axes.draw_artist if artist.axes else fig.draw_artist)(artist)
        frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[...,:3].copy())
        frames.append(frame)
    plt.close(fig)
    return frames, dict(rollout_index=rollout, scene_ADE_3d_m=float(scores[rollout]), zoom=zoom,
                        bounds_enu_m=bounds.tolist(), frames=len(frames), skip=skip)


def save_gif(frames, path, skip):
    # One palette across time keeps the static basemap identical in GIF deltas.
    atlas = Image.new('RGB', (600, 450*3))
    for row, index in enumerate((0, len(frames)//2, len(frames)-1)):
        atlas.paste(frames[index].convert('RGB').resize((600,450)), (0,row*450))
    palette = atlas.quantize(colors=128, dither=Image.Dither.NONE)
    frames = [frame.convert('RGB').quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    durations = [round(1000/9)*skip]*len(frames)
    durations[-1] = 1500
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=True, disposal=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root.resolve(); out = root/'outputs/demo_gif'; out.mkdir(parents=True, exist_ok=True)
    records, thumbs = [], []
    for path in select_windows(root):
        data = load_rollout(path)
        frames, record = render_window(data, root/'data/basemap_cache')
        dest = out/f'{data["scenario"]}_f{int(data["start_frame"]):04d}.gif'
        save_gif(frames, dest, 1)
        if dest.stat().st_size > 8_000_000:
            frames = frames[::2]; record.update(frames=len(frames), skip=2)
            save_gif(frames, dest, 2)
        if dest.stat().st_size > 8_000_000:
            raise RuntimeError(f'GIF exceeds 8 MB: {dest}')
        thumb = frames[-1].convert('RGB'); thumb.thumbnail((480,360)); thumbs.append(thumb)
        record.update(input=str(path.relative_to(root)), output=dest.name, bytes=dest.stat().st_size,
                      input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        records.append(record)
        print(json.dumps(record), flush=True)
    sheet = Image.new('RGB', (2400,720), '#f5f8f7')
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i%5)*480, (i//5)*360))
    sheet.save(out/'contact_sheet.png')
    (out/'manifest.json').write_text(json.dumps(dict(seed=0, draft='v1', selection='six existing + four random >=3 valid aircraft; no error ranking',
        score='equal-aircraft mean 3D ADE over all valid future frames of targets', windows=records), indent=2)+'\n')


if __name__ == '__main__':
    main()
