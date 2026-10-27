"""Three-way camera-conditioned forecast demo from saved, separate input/GT arrays."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Affine2D
import matplotlib.patheffects as pe
import numpy as np
from PIL import Image

from src.eval import demo_gif as visual
from src.eval.anomaly_gif import DRONE

COLOURS={'Camera tracker':'#4cc9f0','Prediction':'#ffd166','Ground truth':'#80ed99'}


def render_camera_scene(data,cache,snapshot=False,dpi=100):
    """Colors identify sources, never arbitrary error-selected prediction modes."""
    camera=data['camera_position'].copy();camera[~data['camera_valid']]=np.nan
    gt=data['gt_position'].copy();gt[~data['gt_valid']]=np.nan
    assert data['prediction'].shape==(1,90,3)
    assert data['prediction_heading'].shape==(1,90)
    prediction=data['prediction'][0].copy()  # Sample0, no scoring/selection.
    # Plot raw camera forecast anchor, followed by the unchanged model outputs.
    pred=np.full_like(camera,np.nan);pred[18]=camera[18];pred[19:]=prediction
    sources={'Camera tracker':camera,'Prediction':pred,'Ground truth':gt}
    points=np.concatenate([p[np.isfinite(p[:,:2]).all(-1),:2] for p in sources.values()])
    bounds=visual.view_bounds(points)
    basemap,zoom,pb=visual.basemap(bounds,cache,'dark')
    palette=visual.fixed_palette(basemap)
    style=visual.THEMES['dark'];marker=DRONE if str(data['kind'])=='drone' else visual.PLANE
    fig=plt.figure(figsize=(12,9),dpi=dpi,facecolor=style['background'])
    ax=fig.add_axes((.025,.09,.95,.81))
    ax.imshow(basemap,extent=(pb[0,0],pb[1,0],pb[1,1],pb[0,1]))
    ax.set(xlim=(pb[0,0],pb[1,0]),ylim=(pb[1,1],pb[0,1]),aspect='equal');ax.axis('off')
    fig.text(.035,.963,'SYGARD  /  CAMERA-BASED FORECAST',fontsize=17,fontweight='bold',color=style['text'])
    fig.text(.035,.928,f'{str(data["kind"]).title()} demo  ·  Camera track {int(data["tracker_id"]):02d}  ·  O’Hare International Airport',fontsize=10,color=style['muted'])
    clock=fig.text(.965,.963,'',ha='right',fontsize=13,fontfamily='monospace',color=style['text'],animated=True)
    handles=[Line2D([],[],color=colour,lw=2.5,ls='--' if name=='Prediction' else (':' if name=='Ground truth' else '-'),label=name) for name,colour in COLOURS.items()]
    fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.025,.047),ncol=3,frameon=False,
        labelcolor=style['text'],fontsize=10,handlelength=3)
    coverage=float(data['camera_future_coverage_s'])
    fig.text(.035,.033,f'6 s camera history → 30 s forecast  ·  One sampled rollout  ·  Camera available for {coverage:.1f} s of forecast',fontsize=9,color=style['text'])
    fig.text(.035,.013,'Camera = tracker estimates, including Kalman predictions. Ground truth is shown only for comparison.',fontsize=8,color=style['muted'])
    fig.text(.975,.013,'© OpenStreetMap contributors',ha='right',fontsize=8,color=style['muted'])
    artists=[clock];tracks=[]
    for name,values in sources.items():
        xy=visual.enu_to_pixel(values[:,:2],zoom);colour=COLOURS[name]
        line,=ax.plot([],[],color=colour,lw=2.8 if name!='Camera tracker' else 3.3,
            ls=(0,(6,4)) if name=='Prediction' else ((0,(1.5,3)) if name=='Ground truth' else '-'),animated=True,
            path_effects=[pe.Stroke(linewidth=5.2,foreground=style['background']),pe.Normal()])
        icon,=ax.plot([],[],marker=marker,ms=24,mfc=colour if name=='Camera tracker' else style['background'],
            mec=colour,mew=1.6,ls='',animated=True)
        label=ax.text(0,0,'',color=style['text'],fontsize=9.5,animated=True,
            bbox=dict(boxstyle='round,pad=.3',fc=style['background'],ec=colour,lw=1.1))
        leader,=ax.plot([],[],color=colour,lw=.8,alpha=.7,animated=True)
        tracks.append((name,values,xy,line,icon,label,leader));artists.extend([line,leader,icon,label])
    fig.canvas.draw();background=fig.canvas.copy_from_bbox(fig.bbox)
    frames=[];times=[108] if snapshot else range(109)
    for t in times:
        fig.canvas.restore_region(background)
        clock.set_text(f'{(t-18)/3:+.1f} s')
        occupied=[];scale=dpi/100
        for name,values,xy,line,icon,label,leader in tracks:
            line.set_data(*xy[:t+1].T)
            observed=np.flatnonzero(np.isfinite(xy[:t+1]).all(-1))
            visible=len(observed)>0
            for artist in [icon,label,leader]:artist.set_visible(visible)
            if not visible:continue
            last=int(observed[-1]);point=xy[last];screen=ax.transData.transform(point)
            previous=observed[-2] if len(observed)>1 else last
            delta=values[last,:2]-values[previous,:2]
            angle=float(np.arctan2(delta[1],delta[0]))
            if name=='Prediction' and last>=19:angle=float(data['prediction_heading'][0,last-19])
            icon.set_data([point[0]],[point[1]]);icon.set_marker(marker.transformed(Affine2D().rotate(angle)))
            text=name
            if last<t:text+=f' (last {(last-18)/3:+.1f}s)'
            label.set_text(text)
            width=(len(text)*6+14)*scale;height=24*scale
            candidates=([(22,-42),(-22,-42),(22,-74),(-22,-74),(22,20)] if name=='Prediction'
                else [(22,20),(-22,20),(22,52),(-22,52),(22,-42),(-22,-42),(22,-74)])
            chosen=None
            for dx,dy in candidates:
                dx*=scale;dy*=scale
                left=screen[0]+dx if dx>0 else screen[0]+dx-width
                bottom=screen[1]+dy
                rect=(left,bottom,left+width,bottom+height)
                fits=rect[0]>ax.bbox.xmin and rect[2]<ax.bbox.xmax and rect[1]>ax.bbox.ymin and rect[3]<ax.bbox.ymax
                overlap=any(rect[0]<r[2] and rect[2]>r[0] and rect[1]<r[3] and rect[3]>r[1] for r in occupied)
                if fits and not overlap:chosen=(rect,dx,dy);break
            if chosen is None:chosen=(rect,dx,dy)
            rect,dx,dy=chosen;occupied.append(rect)
            label.set_ha('left' if dx>0 else 'right');label.set_va('bottom')
            target=ax.transData.inverted().transform(screen+[dx,dy]);label.set_position(target)
            leader.set_data([point[0],target[0]],[point[1],target[1]])
        for artist in artists:(artist.axes.draw_artist if artist.axes else fig.draw_artist)(artist)
        rgb=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[...,:3].copy())
        frames.append(rgb if snapshot else rgb.quantize(palette=palette,dither=Image.Dither.NONE))
    plt.close(fig)
    return frames,dict(frames=len(frames),size=list(frames[0].size),bounds_enu_m=bounds.tolist(),zoom=zoom,
        prediction_sample_index=0,history_s=6,forecast_s=30,camera_future_coverage_s=coverage)
