"""GT-only animation with saved M8 NLL; no model loading or inference."""
import argparse
from bisect import bisect_right
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
import matplotlib.patheffects as pe
import numpy as np
from PIL import Image

from src.eval import demo_gif as visual
from src.eval.anomaly_score import load_drone, DRONE_PATH
from src.eval.load_rollout import load_rollout

ALERT = '#ff3038'
# Avoid red as a normal track colour, so the anomaly outline has one meaning.
TRACK_COLOURS = [visual.PALETTE[i] for i in [2, 1, 3, 4, 5, 7, 8]]


def read_scores(path):
    tracks = {}
    seen = set()
    with Path(path).open() as stream:
        for row in csv.DictReader(stream):
            key = (row['source'], row['scenario'], row['agent_id'])
            start = int(row['token_start_frame'])
            unique = (*key, start)
            value = float(row['nll'])
            if unique in seen or not np.isfinite(value):
                raise ValueError('Duplicate or nonfinite saved score')
            seen.add(unique)
            tracks.setdefault(key, []).append(dict(start=start, end=start+6, nll=value))
    for rows in tracks.values():
        rows.sort(key=lambda row: row['end'])
    return tracks


def completed_score(rows, frame):
    """Latest completed token, valid for <2s; never bridge an unscored gap."""
    index = bisect_right([r['end'] for r in rows], frame)-1
    if index < 0 or frame-rows[index]['end'] >= 6:
        return None
    return rows[index]


def high_segments(xy, valid, frame_offset, rows, threshold):
    """Only the six GT edges belonging to a high token receive its halo."""
    segments, completions = [], []
    for row in rows:
        if row['nll'] <= threshold:
            continue
        first = max(1, row['start']+1-frame_offset)
        last = min(len(xy)-1, row['end']-frame_offset)
        for end in range(first, last+1):
            if valid[end-1:end+1].all():
                segments.append(xy[end-1:end+1])
                completions.append(row['end'])
    return np.asarray(segments).reshape(-1,2,2), np.asarray(completions)


def render_scene(data, tracks, cache, threshold=7.):
    source, scenario = str(data['source']), str(data['scenario'])
    offset = int(data['start_frame'])
    valid = data['valid_mask']
    gt = data['gt_position'][..., :2].astype(float).copy()
    gt[~valid] = np.nan
    bounds = visual.view_bounds(gt[valid])
    image, zoom, pb = visual.basemap(bounds, cache, 'dark')
    palette = visual.fixed_palette(image, extra_colours=[ALERT])
    pixels = visual.enu_to_pixel(gt, zoom)
    style = visual.THEMES['dark']
    fig = plt.figure(figsize=(12,9), dpi=100, facecolor=style['background'])
    ax = fig.add_axes(visual.MAP_RECT)
    ax.imshow(image, extent=(pb[0,0],pb[1,0],pb[1,1],pb[0,1]))
    ax.set(xlim=(pb[0,0],pb[1,0]), ylim=(pb[1,1],pb[0,1]), aspect='equal')
    ax.axis('off')
    fig.text(.035,.958,f'{data["date"]} · {scenario}',fontsize=16,color=style['text'])
    clock = fig.text(.965,.958,'',ha='right',fontsize=15,fontfamily='monospace',color=style['text'],animated=True)
    fig.text(.035,.032,f'GT replay · offline NLL (single-agent, teacher-forced) · red outline: NLL > {threshold:g}',fontsize=10,color=style['text'])
    fig.text(.035,.012,'Latest completed 2 s GT token. Intervals use scenario time; N/A = no valid score.',fontsize=8,color=style['muted'])
    fig.text(.975,.015,'© OpenStreetMap contributors',ha='right',fontsize=8,color=style['muted'])
    colours = {agent:TRACK_COLOURS[i%len(TRACK_COLOURS)] for i,agent in enumerate(sorted(map(str,data['agent_id'])))}
    objects, artists, warnings = [], [clock], []
    for a, agent in enumerate(data['agent_id']):
        if not valid[a].any():
            continue
        colour = colours[str(agent)]
        history, = ax.plot([],[],color=colour,lw=3.,animated=True,
            path_effects=[pe.Stroke(linewidth=5.5,foreground=style['background']),pe.Normal()])
        halo = LineCollection([],colors=ALERT,linewidths=8,capstyle='round',animated=True)
        core = LineCollection([],colors=colour,linewidths=3,capstyle='round',animated=True)
        ax.add_collection(halo); ax.add_collection(core)
        box = Rectangle((0,0),1,1,fill=False,edgecolor=ALERT,linewidth=2.2,animated=True)
        ax.add_patch(box)
        icon, = ax.plot([],[],marker=visual.PLANE,ms=25,mfc=colour,mec=style['background'],mew=1.1,ls='',animated=True)
        label = ax.text(0,0,'',fontsize=10,color=style['text'],animated=True,
            bbox=dict(facecolor=style['background'],edgecolor=colour,linewidth=1,pad=2))
        rows = tracks.get((source,scenario,str(agent)),[])
        if any(row['start']%6 != offset%6 for row in rows):
            warnings.append(f'{agent}: score grid does not align with displayed window; scores omitted without rounding')
            rows=[]
        segments, ends = high_segments(pixels[a],valid[a],offset,rows,threshold)
        objects.append((a,str(agent),rows,history,halo,core,box,icon,label,segments,ends))
        artists += [history,halo,core,box,icon,label]
    fig.canvas.draw(); background = fig.canvas.copy_from_bbox(fig.bbox)
    frames = []
    for local in range(gt.shape[1]):
        absolute = offset+local
        fig.canvas.restore_region(background)
        clock.set_text(f't = {absolute/3:6.1f} s')
        for a,agent,rows,history,halo,core,box,icon,label,segments,ends in objects:
            history.set_data(*pixels[a,:local+1].T)
            shown = segments[ends<=absolute]
            halo.set_segments(shown); core.set_segments(shown)
            visible = bool(valid[a,local])
            icon.set_visible(visible); label.set_visible(visible)
            row = completed_score(rows,absolute) if visible else None
            high = row is not None and row['nll']>threshold
            box.set_visible(bool(visible and high))
            if visible:
                point = pixels[a,local]
                screen = ax.transData.transform(point)
                icon.set_data([point[0]],[point[1]])
                icon.set_marker(visual.PLANE.transformed(Affine2D().rotate(float(data['gt_heading'][a,local]))))
                left,bottom = ax.transData.inverted().transform(screen-[23,23])
                right,top = ax.transData.inverted().transform(screen+[23,23])
                box.set_bounds(left,bottom,right-left,top-bottom)
                right_space = ax.bbox.xmax-screen[0]
                above_space = ax.bbox.ymax-screen[1]
                dx,ha = (20,'left') if right_space>190 else (-20,'right')
                dy,va = (20,'bottom') if above_space>65 else (-20,'top')
                label.set_position(ax.transData.inverted().transform(screen+[dx,dy]))
                label.set_ha(ha); label.set_va(va)
                label.set_text(f'{agent}  NLL {row["nll"]:.2f}\n[{row["start"]/3:g}–{row["end"]/3:g} s]' if row else f'{agent}  NLL N/A')
                label.get_bbox_patch().set_edgecolor(ALERT if high else colours[agent])
                label.get_bbox_patch().set_linewidth(2 if high else 1)
        for artist in artists:
            (artist.axes.draw_artist if artist.axes else fig.draw_artist)(artist)
        rgb = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[...,:3].copy())
        frames.append(rgb.quantize(palette=palette,dither=Image.Dither.NONE))
    plt.close(fig)
    return frames,dict(source=source,scenario=scenario,start_frame=offset,frames=len(frames),
        threshold=threshold,objects=len(objects),bounds_enu_m=bounds.tolist(),zoom=zoom,warnings=warnings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/anomaly/gifs'))
    parser.add_argument('--threshold',type=float,default=7.)
    args = parser.parse_args(); root=args.root.resolve(); output=root/args.output_dir
    output.mkdir(parents=True,exist_ok=True)
    score_file=root/'outputs/anomaly/token_scores.csv'
    score_hash=hashlib.sha256(score_file.read_bytes()).hexdigest()
    tracks=read_scores(score_file); drone,drone_info=load_drone(DRONE_PATH)
    drone_scene=dict(source='drone',scenario='ord_drone_2026-09-28_1800z_3hz',date=str(drone['date']),
        start_frame=0,gt_position=drone['pos'],gt_heading=drone['heading'],valid_mask=drone['valid'],agent_id=drone['agent_id'])
    scenes=[(drone_scene,DRONE_PATH,'drone01_full')]
    for path in visual.select_windows(root):
        saved=load_rollout(path)
        saved['source']='aircraft'
        scenes.append((saved,path,f'{saved["scenario"]}_f{int(saved["start_frame"]):04d}'))
    records,thumbs=[],[]
    for data,path,name in scenes:
        frames,record=render_scene(data,tracks,root/'data/basemap_cache',args.threshold)
        destination=output/f'{name}.gif'
        visual.save_gif(frames,destination,1);skip=1
        if destination.stat().st_size>8_000_000:
            frames=frames[::2];skip=2;visual.save_gif(frames,destination,skip)
        if destination.stat().st_size>8_000_000:
            raise RuntimeError(f'GIF exceeds8MB: {destination}')
        with Image.open(destination) as decoded:
            decoded.seek(decoded.n_frames-1);thumb=decoded.convert('RGB')
        thumb.thumbnail((480,360));thumbs.append(thumb)
        if data['source']=='drone':
            # Show a low-score moment and the first observed high-score moment.
            rows=tracks[('drone',str(data['scenario']),'drone01')]
            high=next((row for row in rows if row['nll']>args.threshold),None)
            after=next((row for row in rows if high and row['end']>high['end'] and row['nll']<=args.threshold),None)
            selected=[rows[0],high,after]
            for label,row in zip(['low','high','low_after_high'],selected):
                if row is None:
                    continue
                with Image.open(destination) as decoded:
                    decoded.seek(row['end']//skip);decoded.convert('RGB').save(output/f'drone_{label}.png')
        record.update(output=destination.name,encoded_frames=len(frames),skip=skip,bytes=destination.stat().st_size,
            input=str(path),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        records.append(record); print(json.dumps(record),flush=True)
    sheet=Image.new('RGB',(1920,1080),visual.THEMES['dark']['background'])
    for i,thumb in enumerate(thumbs):
        sheet.paste(thumb,((i%4)*480,(i//4)*360))
    sheet.save(output/'contact_sheet.png')
    assert score_hash==hashlib.sha256(score_file.read_bytes()).hexdigest()
    manifest=dict(score_file=str(score_file.relative_to(root)),score_sha256=score_hash,
        threshold=args.threshold,semantics='latest completed 2s GT token; score expires after2s without a new valid token; high trail appears only after completion',
        nll_is_existing_m8_score=True,model_execution=False,training=False,drone_interpolation=drone_info,gifs=records)
    manifest['drone_high_intervals']=[dict(start_s=row['start']/3,end_s=row['end']/3,nll=row['nll'])
        for row in tracks[('drone',drone_scene['scenario'],'drone01')] if row['nll']>args.threshold]
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    main()
