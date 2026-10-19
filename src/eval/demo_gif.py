"""Colour-preserving top-down animations from saved rollouts; never invokes a model."""
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
import matplotlib.patheffects as pe
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import requests

from src.eval.load_rollout import load_rollout

ORIGIN = (41.980103, -87.903868)
RADIUS = 6378137.
SIZE = (1200, 900)
MAP_RECT = (.025, .055, .95, .86)
PALETTE = ['#ff6b6b', '#ffd166', '#4cc9f0', '#80ed99', '#c77dff',
           '#ff9f1c', '#f72585', '#3a86ff', '#2ec4b6', '#fca311']
THEMES = dict(dark=dict(background='#0b1622', text='#e8eef5', muted='#8fa3b8'),
              light=dict(background='#f5f8f7', text='#233e48', muted='#526873'))
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


def basemap(bounds, cache, theme='dark'):
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
    if theme == 'dark':
        faded = ImageOps.colorize(ImageOps.invert(ImageOps.grayscale(crop)), '#0b1622', '#55677d')
        faded = ImageEnhance.Contrast(faded).enhance(.8)
    else:
        faded = Image.blend(ImageEnhance.Color(crop).enhance(.45), Image.new('RGB', crop.size, 'white'), .45)
    return faded, zoom, np.array([[left, top], [right, bottom]])


def rgb(color):
    return np.array(matplotlib.colors.to_rgb(color))


def track_colours(color, theme):
    base = rgb(color)
    return base, base*.55+.45, base*.25+rgb(THEMES[theme]['background'])*.75


def fixed_palette(map_image, theme='dark', extra_colours=()):
    """Reserve trajectory colours and antialias blends; map gets remaining slots."""
    style = THEMES[theme]
    colours = [tuple(round(x*255) for x in rgb(style[key])) for key in ('background','text','muted')]
    colours.append((255,255,255))
    for color in PALETTE:
        for variant in track_colours(color, theme):
            for weight in (1., .75, .5, .25):
                mixed = variant*weight+rgb(style['background'])*(1-weight)
                colours.append(tuple(np.rint(mixed*255).astype(int)))
    for color in extra_colours:
        for weight in (1., .75, .5, .25):
            mixed = rgb(color)*weight+rgb(style['background'])*(1-weight)
            colours.append(tuple(np.rint(mixed*255).astype(int)))
    remaining = 255-len(colours)  # Index255 is reserved solely for delta transparency.
    mapped = map_image.convert('RGB').quantize(colors=remaining, dither=Image.Dither.NONE)
    raw = mapped.getpalette()
    colours += [tuple(raw[i*3:i*3+3]) for i in range(remaining)]
    colours.append((1,2,3))
    palette = Image.new('P',(1,1))
    palette.putpalette([component for color in colours for component in color])
    return palette


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


def render_window(data, cache, skip=1, theme='dark', snapshot=False, presentation=False, dpi=100):
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
    image, zoom, pb = basemap(bounds, cache, theme)
    palette = fixed_palette(image, theme)
    style = THEMES[theme]
    gtpx, predpx = enu_to_pixel(gt, zoom), enu_to_pixel(pred, zoom)
    fig = plt.figure(figsize=(12, 9), dpi=dpi, facecolor=style['background'])
    ax = fig.add_axes(MAP_RECT)
    ax.imshow(image, extent=(pb[0,0], pb[1,0], pb[1,1], pb[0,1]))
    ax.set(xlim=(pb[0,0], pb[1,0]), ylim=(pb[1,1], pb[0,1]), aspect='equal')
    ax.axis('off')
    if presentation:
        from matplotlib.lines import Line2D
        fig.text(.035,.965,'SYGARD  /  MOTION SIMULATION',fontsize=18,weight='bold',color=style['text'])
        fig.text(.035,.932,f'Aircraft trajectory rollout  ·  O’Hare International Airport  ·  {data["date"]}',
            fontsize=10,color=style['muted'])
        handles=[Line2D([],[],color=style['text'],lw=2.5,label='Recorded trajectory'),
            Line2D([],[],color=style['text'],lw=2,ls=(0,(6,4)),label='Simulated trajectory')]
        fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.026,.027),ncol=2,
            frameon=False,fontsize=10,labelcolor=style['text'],handlelength=3)
        fig.text(.035,.014,'6 s observed history  /  30 s forecast  /  Filled aircraft: recorded · Outline: simulated',
            fontsize=8.5,color=style['muted'])
    else:
        fig.text(.035, .955, f'{data["date"]} · {data["scenario"]}', fontsize=18, color=style['text'], weight='medium')
        fig.text(.035, .025, '━━  ground truth      ┄┄  prediction', fontsize=10, color=style['muted'])
    fig.text(.975, .025, '© OpenStreetMap contributors', ha='right', fontsize=9, color=style['muted'])
    clock = fig.text(.965, .955, '', ha='right', fontsize=14 if presentation else 16,
        fontfamily='monospace', color=style['text'], animated=True)
    artists = [clock]
    tracks = []
    colour_index = {str(agent): i % len(PALETTE) for i, agent in enumerate(sorted(map(str,data['agent_id'])))}
    for a, agent in enumerate(data['agent_id']):
        if not valid[a].any():
            continue
        color, tint, faint = track_colours(PALETTE[colour_index[str(agent)]], theme)
        ax.plot(*gtpx[a].T, color=faint, lw=1.5)
        truth, = ax.plot([], [], color=color, lw=3.5, animated=True,
                        path_effects=[pe.Stroke(linewidth=6,foreground=style['background']),pe.Normal()])
        future, = ax.plot([], [], color=tint, lw=2.6, ls=(0,(6,4)), animated=True)
        icon, = ax.plot([], [], marker=PLANE, ms=26, mfc=color, mec=style['background'], mew=1.2, ls='', animated=True)
        ghost, = ax.plot([], [], marker=PLANE, ms=22, mfc=style['background'], mec=tint, mew=2, ls='', animated=True)
        label = ax.text(0, 0, f'Aircraft {colour_index[str(agent)]+1:02d}' if presentation else str(agent), fontsize=11, color=style['text'], animated=True,
                        bbox=dict(facecolor=style['background'], edgecolor=color, linewidth=1, pad=2))
        tracks.append((a, truth, future, icon, ghost, label))
        artists.extend([truth, future, ghost, icon, label])
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(fig.bbox)
    frames = []
    for t in ([108] if snapshot else frame_indices(skip)):
        fig.canvas.restore_region(background)
        clock.set_text('FORECAST  +30 s' if presentation and snapshot else f't = {(t-18)/3:+.1f} s')
        for a, truth, future, icon, ghost, label in tracks:
            truth.set_data(*gtpx[a, :t+1].T)
            k = max(0, t-18)
            trail = np.concatenate((gtpx[a, 18:19], predpx[a, :k])) if k else np.empty((0, 2))
            future.set_data(*trail.T)
            icon.set_visible(bool(valid[a,t])); label.set_visible(bool(valid[a,t]))
            if valid[a,t]:
                icon.set_data([gtpx[a,t,0]], [gtpx[a,t,1]])
                icon.set_marker(PLANE.transformed(Affine2D().rotate(float(data['gt_heading'][a,t]))))
                # Convert a 16-display-pixel offset to map coordinates.
                screen=ax.transData.transform(gtpx[a,t])
                scale=dpi/100
                dx=16*scale if not presentation or ax.bbox.xmax-screen[0]>150*scale else -16*scale
                dy=16*scale if not presentation or ax.bbox.ymax-screen[1]>45*scale else -16*scale
                label.set_position(ax.transData.inverted().transform(screen+[dx,dy]))
                label.set_ha('left' if dx>0 else 'right')
                label.set_va('baseline' if dy>0 else 'top')
            show_pred = k > 0 and pred_valid[a,k-1]
            ghost.set_visible(bool(show_pred))
            if show_pred:
                ghost.set_data([predpx[a,k-1,0]], [predpx[a,k-1,1]])
                ghost.set_marker(PLANE.transformed(Affine2D().rotate(float(data['heading'][rollout,a,k-1]))))
        for artist in artists:
            (artist.axes.draw_artist if artist.axes else fig.draw_artist)(artist)
        frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[...,:3].copy())
        frames.append(frame if snapshot else frame.quantize(palette=palette, dither=Image.Dither.NONE))
    plt.close(fig)
    return frames, dict(rollout_index=rollout, scene_ADE_3d_m=float(scores[rollout]), zoom=zoom,
                        bounds_enu_m=bounds.tolist(), frames=len(frames), skip=skip, theme=theme,
                        snapshot=snapshot,presentation=presentation,final_time_s=30 if snapshot else None,
                        aircraft_colours={str(a):PALETTE[colour_index[str(a)]] for a in data['agent_id']})


def save_gif(frames, path, skip):
    # Explicit transparent deltas compress the static map without re-indexing
    # the exact reserved colours (Pillow's optimize=True can re-index palettes).
    palette = frames[0].getpalette()
    encoded, previous = [], None
    for frame in frames:
        if frame.mode != 'P' or frame.getpalette() != palette:
            raise ValueError('Frames must share the reserved palette')
        indices = np.asarray(frame).copy()
        indices[indices == 255] = 0
        delta = indices.copy()
        if previous is not None:
            delta[indices == previous] = 255
        converted = Image.fromarray(delta, mode='P')
        converted.putpalette(palette)
        encoded.append(converted)
        previous = indices
    durations = [round(1000/9)*skip]*len(frames)
    durations[-1] = 1500
    encoded[0].save(path, save_all=True, append_images=encoded[1:], duration=durations,
                   loop=0, optimize=False, disposal=1, transparency=255)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/demo_gif'))
    parser.add_argument('--theme', choices=THEMES, default='dark')
    args = parser.parse_args()
    root = args.root.resolve(); out = root/args.output_dir; out.mkdir(parents=True, exist_ok=True)
    records, thumbs = [], []
    for path in select_windows(root):
        data = load_rollout(path)
        frames, record = render_window(data, root/'data/basemap_cache', theme=args.theme)
        dest = out/f'{data["scenario"]}_f{int(data["start_frame"]):04d}.gif'
        save_gif(frames, dest, 1)
        if dest.stat().st_size > 8_000_000:
            frames = frames[::2]; record.update(frames=len(frames), skip=2)
            save_gif(frames, dest, 2)
        if dest.stat().st_size > 8_000_000:
            raise RuntimeError(f'GIF exceeds 8 MB: {dest}')
        with Image.open(dest) as decoded:
            decoded.seek(decoded.n_frames-1)
            thumb = decoded.convert('RGB')
        thumb.thumbnail((480,360)); thumbs.append(thumb)
        record.update(input=str(path.relative_to(root)), output=dest.name, bytes=dest.stat().st_size,
                      input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        records.append(record)
        print(json.dumps(record), flush=True)
    sheet = Image.new('RGB', (2400,720), THEMES[args.theme]['background'])
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i%5)*480, (i//5)*360))
    sheet.save(out/'contact_sheet.png')
    (out/'manifest.json').write_text(json.dumps(dict(seed=0, draft='v2', theme=args.theme, contact_sheet_source='decoded GIF last frames', selection='six existing + four random >=3 valid aircraft; no error ranking',
        score='equal-aircraft mean 3D ADE over all valid future frames of targets', windows=records), indent=2)+'\n')


if __name__ == '__main__':
    main()
