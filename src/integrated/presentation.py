"""Read-only integrated demo presentation, from accepted D4 arrays/evidence."""
import base64
import io
import json
import math
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon,FancyBboxPatch,Rectangle
from matplotlib.transforms import Affine2D
import matplotlib.patheffects as pe
import numpy as np
from PIL import Image,ImageOps,ImageEnhance

from src.eval import demo_gif as visual
from src.eval.anomaly_gif import DRONE
from src.integrated.scenario_status import current_only_timeline

BG='#0b1622';TEXT='#e8eef5';MUTED='#8fa3b8';CYAN='#4cc9f0';AMBER='#ffd166';GREEN='#80ed99';PURPLE='#c77dff'
ORANGE='#ff9f43'
LEVEL_COLOURS={'Normal traffic':CYAN,'Unusual activity':ORANGE,'Potential intrusion':'#ff626b',
    'Intrusion detected':'#ff626b','Collecting observations':MUTED}


def cached_basemap(bounds,cache):
    """Strictly offline mosaic; fall back to a cached coarser zoom, no HTTP."""
    span=bounds[1]-bounds[0]
    world=2*np.pi*visual.RADIUS*np.cos(np.radians(visual.ORIGIN[0]))
    preferred=min(18,int(math.floor(math.log2(min(1140/span[0],774/span[1])*world/256))))
    for zoom in range(preferred,-1,-1):
        p=visual.enu_to_pixel(bounds,zoom)
        left,right=sorted(p[:,0]);top,bottom=sorted(p[:,1])
        x0,x1,y0,y1=int(left//256),int(right//256),int(top//256),int(bottom//256)
        paths={(x,y):Path(cache)/str(zoom)/str(x)/f'{y}.png' for x in range(x0,x1+1) for y in range(y0,y1+1)}
        if not all(path.exists() for path in paths.values()):continue
        mosaic=Image.new('RGB',((x1-x0+1)*256,(y1-y0+1)*256))
        for (x,y),path in paths.items():
            with Image.open(path) as tile:mosaic.paste(tile.convert('RGB'),((x-x0)*256,(y-y0)*256))
        crop=mosaic.crop((left-x0*256,top-y0*256,right-x0*256,bottom-y0*256))
        faded=ImageOps.colorize(ImageOps.invert(ImageOps.grayscale(crop)),'#0b1622','#55677d')
        faded=ImageEnhance.Contrast(faded).enhance(.8)
        return faded,zoom,np.array([[left,top],[right,bottom]])
    raise FileNotFoundError('No complete locally cached map zoom; no network requests made')


def load_presentation(root,scene_id):
    directory=Path(root)/'outputs/integrated_demo'/scene_id
    manifest=json.loads((directory/'manifest.json').read_text())
    timeline=json.loads((directory/'timeline.json').read_text())
    records=json.loads((directory/'forecasts.json').read_text());forecasts={}
    for r in records:
        with np.load(Path(root)/'outputs/integrated_demo'/r['prediction_file']) as a:
            # CV is deliberately not read into presentation.
            forecasts[r['frame']]=dict(position=a['position'].copy(),heading=a['heading'].copy(),medoid=int(a['medoid']))
    with np.load(directory/'observations.npz') as a:stream={key:a[key].copy() for key in a.files}
    timeline=current_only_timeline(timeline,stream['observed'],manifest)
    points=[stream['observed'][...,:2].reshape(-1,2),np.array(manifest['zone']['polygon_enu_m'])]
    points += [p['position'][p['medoid'],...,:2].reshape(-1,2) for p in forecasts.values()]
    bounds=visual.view_bounds(np.concatenate(points),aspect=(16*.68)/(9*.77))
    image,zoom,pb=cached_basemap(bounds,Path(root)/'data/basemap_cache')
    return dict(manifest=manifest,timeline=timeline,forecasts=forecasts,stream=stream,
        map_image=image,zoom=zoom,pb=pb,bounds=bounds)


def display_texts(row,manifest):
    scenario=row['scenario']
    reasons='\n\n'.join(textwrap.fill(r,width=34) for r in scenario['reasons'])
    objects=[]
    for i,o in enumerate(row['objects']):
        name='Drone 01' if i==0 else 'Aircraft 01'
        valid=o['current_state_nll'] is not None and np.isfinite(o['current_state_nll'])
        status='Unusual behavior' if o['anomalous'] else ('Normal traffic' if valid else 'Collecting observations')
        message=f'{name} · {status}'
        if o['anomalous']:message+='\n'+textwrap.fill(o['anomaly_reason'],width=34)
        objects.append(message)
    return dict(scenario=textwrap.fill(scenario['status'],width=19),explanation=reasons,objects='\n\n'.join(objects))


class DemoRenderer:
    def __init__(self,data,dpi=100):
        self.data=data;self.dpi=dpi;self.zoom=data['zoom']
        m=data['manifest'];scene=m['scene'];self.scene=scene
        self.fig=plt.figure(figsize=(16,9),dpi=dpi,facecolor=BG)
        self.ax=self.fig.add_axes((.025,.13,.68,.75));pb=data['pb']
        self.ax.imshow(data['map_image'],extent=(pb[0,0],pb[1,0],pb[1,1],pb[0,1]))
        self.ax.set(xlim=(pb[0,0],pb[1,0]),ylim=(pb[1,1],pb[0,1]),aspect='equal');self.ax.axis('off')
        self.fig.text(.027,.955,'SYGARD  /  AIRSPACE MONITOR',fontsize=21,fontweight='bold',color=TEXT)
        title='DAYTIME HOBBYIST' if scene['context']=='daytime' else 'NIGHTTIME APPROACH'
        sigma=scene['scripted_noise_sigma_m_per_axis']
        self.fig.text(.027,.91,f'{title}  ·  SCRIPTED MVP DEMO  ·  Simulated tracker σ={sigma:g} m / axis',fontsize=11,color=MUTED)
        self.clock=self.fig.text(.70,.955,'',ha='right',fontsize=17,fontfamily='monospace',color=TEXT,animated=True)
        zone=visual.enu_to_pixel(np.array(m['zone']['polygon_enu_m']),self.zoom)
        self.ax.add_patch(Polygon(zone,closed=True,facecolor=PURPLE,edgecolor=PURPLE,alpha=.22,lw=2))
        center=zone.mean(0);self.ax.text(*center,'DEMO\nPROTECTED ZONE',ha='center',va='center',color='#e4c7ff',fontsize=9,fontweight='bold')
        self.fig.text(.03,.085,'━━ Observed tracker     ┄┄ Model forecast     □ Current anomaly',fontsize=11,color=TEXT)
        self.fig.text(.03,.054,'32 joint sampled futures · GT-free medoid shown · updates every 2 s · 30 s forecast',fontsize=10,color=MUTED)
        self.fig.text(.03,.028,'Orange: current object anomaly. Scenario alert: anomalous object inside or forecast to enter the protected zone.',fontsize=9,color=MUTED)
        self.fig.text(.70,.012,'© OpenStreetMap contributors',fontsize=8,color=MUTED,ha='right')
        self.cards={}
        for key,y,h,title in [('scenario',.69,.19,'SCENARIO STATUS'),
            ('explanation',.37,.29,'WHY THIS STATUS'),('objects',.045,.29,'OBJECT STATUS')]:
            self.fig.add_artist(FancyBboxPatch((.735,y),.245,h,boxstyle='round,pad=.007',
                transform=self.fig.transFigure,fc='#132434',ec='#294052',lw=1))
            self.fig.text(.75,y+h-.028,title,color=MUTED,fontsize=9,fontweight='bold')
            text=self.fig.text(.75,y+h-.064,'',color=TEXT,fontsize=11 if key!='scenario' else 21,
                va='top',animated=True,linespacing=1.5)
            self.cards[key]=text
        self.tracks=[];self.boxes=[];self.artists=[self.clock,*self.cards.values()]
        self.observed=visual.enu_to_pixel(data['stream']['observed'][...,:2],self.zoom)
        for agent,(colour,marker,label) in enumerate([(CYAN,DRONE,'Drone 01'),(GREEN,visual.PLANE,'Aircraft 01')]):
            history,=self.ax.plot([],[],color=colour,lw=3 if agent==0 else 2,animated=True,
                path_effects=[pe.Stroke(linewidth=5,foreground=BG),pe.Normal()])
            future,=self.ax.plot([],[],color=AMBER if agent==0 else GREEN,lw=2.8 if agent==0 else 1.5,
                ls=(0,(6,4)),alpha=1 if agent==0 else .65,animated=True)
            icon,=self.ax.plot([],[],marker=marker,ms=24,mfc=colour,mec=BG,mew=1,ls='',animated=True)
            endpoint,=self.ax.plot([],[],marker='D',ms=7,mfc=AMBER if agent==0 else GREEN,mec=BG,ls='',animated=True)
            text=self.ax.text(0,0,label,color=TEXT,fontsize=10,animated=True,
                bbox=dict(fc=BG,ec=colour,pad=2))
            box=Rectangle((0,0),1,1,fill=False,edgecolor=ORANGE,lw=2.5,animated=True,visible=False)
            self.ax.add_patch(box);self.boxes.append(box);self.artists.append(box)
            self.tracks.append((history,future,icon,endpoint,text,marker));self.artists += [history,future,icon,endpoint,text]
        self.fig.canvas.draw();self.background=self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def frame(self,frame):
        data=self.data;row=data['timeline'][frame];texts=display_texts(row,data['manifest'])
        self.fig.canvas.restore_region(self.background);self.clock.set_text(f"t = {row['time_s']:5.1f} s")
        for key,text in self.cards.items():text.set_text(texts[key])
        self.cards['scenario'].set_color(LEVEL_COLOURS[row['scenario']['status']])
        origin=row['objects'][0]['forecast_origin_frame'];forecast=data['forecasts'].get(origin)
        for agent,(history,future,icon,endpoint,text,marker) in enumerate(self.tracks):
            points=self.observed[agent];point=points[frame];history.set_data(*points[:frame+1].T)
            icon.set_data([point[0]],[point[1]])
            delta=data['stream']['observed'][agent,frame,:2]-data['stream']['observed'][agent,max(0,frame-6),:2]
            icon.set_marker(marker.transformed(Affine2D().rotate(float(np.arctan2(delta[1],delta[0])))))
            obj=row['objects'][agent];anomalous=obj['anomalous']
            label='Drone 01' if agent==0 else 'Aircraft 01'
            text.set_text(label+(' · Unusual behavior\n'+obj['anomaly_reason'] if anomalous else ''))
            text.get_bbox_patch().set_edgecolor(ORANGE if anomalous else (CYAN if agent==0 else GREEN))
            screen=self.ax.transData.transform(point);scale=self.dpi/100
            corners=self.ax.transData.inverted().transform([screen-20*scale,screen+20*scale])
            box=self.boxes[agent];box.set_xy(corners.min(0));box.set_width(abs(corners[1,0]-corners[0,0]));box.set_height(abs(corners[1,1]-corners[0,1]));box.set_visible(anomalous)
            dx=18*scale if self.ax.bbox.xmax-screen[0]>130*scale else -18*scale
            dy=18*scale if self.ax.bbox.ymax-screen[1]>35*scale else -28*scale
            text.set_position(self.ax.transData.inverted().transform(screen+[dx,dy]));text.set_ha('left' if dx>0 else 'right')
            if forecast is not None:
                remaining=forecast['position'][forecast['medoid'],agent,frame-origin:]
                projected=visual.enu_to_pixel(remaining[:,:2],self.zoom)
                path=np.vstack([point,projected]);future.set_data(*path.T)
                endpoint.set_data([projected[-1,0]],[projected[-1,1]])
            else:future.set_data([],[]);endpoint.set_data([],[])
        for artist in self.artists:(artist.axes.draw_artist if artist.axes else self.fig.draw_artist)(artist)
        return Image.fromarray(np.asarray(self.fig.canvas.buffer_rgba())[...,:3].copy())

    def close(self):plt.close(self.fig)


def viewer_bundle(data):
    buffer=io.BytesIO();data['map_image'].save(buffer,format='PNG')
    result=dict(map='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode(),
        pb=data['pb'].tolist(),zone=visual.enu_to_pixel(np.array(data['manifest']['zone']['polygon_enu_m']),data['zoom']).tolist(),
        manifest=data['manifest'],timeline=data['timeline'],forecasts={},
        observed=visual.enu_to_pixel(data['stream']['observed'][...,:2],data['zoom']).round(3).tolist(),
        reference=visual.enu_to_pixel(data['stream']['reference'][...,:2],data['zoom']).round(3).tolist())
    for frame,forecast in data['forecasts'].items():
        result['forecasts'][frame]=dict(medoid=forecast['medoid'],
            xy=visual.enu_to_pixel(forecast['position'][...,:2],data['zoom']).round(3).tolist())
    return result
