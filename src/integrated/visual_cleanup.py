"""D10 presentation only: retain D9 data, extents, grouping and incident rules."""
import numpy as np
from PIL import Image
from matplotlib.transforms import Affine2D
import matplotlib.patheffects as pe
from src.eval import demo_gif as visual
from src.integrated.presentation import DemoRenderer,BG,TEXT,MUTED,ORANGE,CYAN,GREEN,AMBER,LEVEL_COLOURS


def integer_shares(groups):
    """Largest remainder, deterministic ties in group order; no weight mutation."""
    if not groups:return []
    total=groups[0]['total_samples']
    if total<=0 or any(g['total_samples']!=total for g in groups) or sum(g['count'] for g in groups)!=total:
        raise ValueError('Groups must partition one sample set')
    numerators=[100*g['count'] for g in groups]
    result=[n//total for n in numerators]
    order=sorted(range(len(groups)),key=lambda i:(-(numerators[i]%total),i))
    for i in order[:100-sum(result)]:result[i]+=1
    return result


def intersects(a,b):
    return a[0]<b[2] and a[2]>b[0] and a[1]<b[3] and a[3]>b[1]


def place_label(anchor,size,bounds,occupied,scale=1.):
    """Return a non-overlapping lower-left screen box, or fail visibly."""
    width,height=size;pad=4*scale;x,y=anchor
    candidates=[]
    for dx in [14*scale,-width-14*scale,36*scale,-width-36*scale,70*scale,-width-70*scale,110*scale,-width-110*scale,170*scale,-width-170*scale]:
        for dy in [14*scale,-height-14*scale,40*scale,-height-40*scale,70*scale,-height-70*scale,110*scale,-height-110*scale,170*scale,-height-170*scale]:
            lx=np.clip(x+dx,bounds[0]+pad,bounds[2]-width-pad)
            ly=np.clip(y+dy,bounds[1]+pad,bounds[3]-height-pad)
            candidates.append(((lx-x)**2+(ly-y)**2,(lx,ly,lx+width,ly+height)))
    def free(box):
        padded=(box[0]-pad,box[1]-pad,box[2]+pad,box[3]+pad)
        return not any(intersects(padded,r) for r in occupied)
    for _,box in sorted(candidates,key=lambda item:item[0]):
        if free(box):return box
    # Whole-map fallback ensures dense endpoints still have nonoverlapping labels.
    candidates=[]
    for ly in np.arange(bounds[1]+pad,bounds[3]-height-pad,max(height+pad,12*scale)):
        for lx in np.arange(bounds[0]+pad,bounds[2]-width-pad,max(width+pad,20*scale)):
            box=(lx,ly,lx+width,ly+height)
            if free(box):candidates.append(((lx-x)**2+(ly-y)**2,box))
    if candidates:return min(candidates,key=lambda item:item[0])[1]
    raise RuntimeError('No collision-free label placement within unchanged map extent')


def cleanup_texts(row):
    """Object lines give state/entry; explanatory reasons occur only in WHY."""
    lines=[]
    for i,obj in enumerate(row['objects']):
        name='Drone 01' if i==0 else f'Aircraft {i:02d}'
        status='Anomalous' if obj['anomalous'] else ('Normal' if obj['current_state_nll'] is not None else 'Pending')
        entry=obj['entry'];share='—' if entry is None else f"{100*entry['entry_samples']/32:g}%"
        lines.append(f'{name} · {status} · zone entry {share}')
    import textwrap
    return dict(scenario=textwrap.fill(row['scenario']['status'],width=19),
        explanation='\n\n'.join(textwrap.fill(r,width=34) for r in row['scenario']['reasons']),
        objects='\n\n'.join(lines))


class CleanupRenderer(DemoRenderer):
    def __init__(self,data,dpi=100):
        super().__init__(data,dpi=dpi)
        # Remove legacy rectangular anomaly artists, rather than merely hiding them.
        for box in self.boxes:self.artists.remove(box);box.remove()
        self.boxes=[]
        for text in list(self.ax.texts):
            if text.get_text()=='DEMO\nPROTECTED ZONE':text.remove()
        for track in self.tracks:
            text=track[4];text.set_bbox(None);text.set_va('bottom');text.set_fontsize(10)
        for _,_,_,label in self.group_artists:
            label.set_va('bottom');label.set_bbox(dict(fc=BG,ec='none',pad=2))
        # Separate aligned columns keep one readable row per object.
        self.cards['objects'].set_visible(False)
        self.object_rows=[]
        for x,title in [(.75,'OBJECT'),(.835,'STATUS'),(.925,'ZONE ENTRY')]:
            self.fig.text(x,.267,title,fontsize=9,color=MUTED,va='top')
        for i in range(len(data['timeline'][0]['objects'])):
            row=[]
            for x in [.75,.835,.925]:
                item=self.fig.text(x,.224-i*.075,'',fontsize=13,color=TEXT,va='top',animated=True)
                row.append(item);self.artists.append(item)
            self.object_rows.append(row)
        # This label is outside the polygon and drawn last over an opaque backing.
        self.zone_label=self.ax.text(0,0,'DEMO PROTECTED ZONE',color='#e4c7ff',fontsize=9,
            fontweight='bold',va='bottom',animated=True,bbox=dict(fc=BG,ec='#785b9a',pad=4))
        zone=visual.enu_to_pixel(np.asarray(data['manifest']['zone']['polygon_enu_m']),self.zoom)
        self.zone_screen=self.ax.transData.transform(zone)
        self.fig.canvas.draw();renderer=self.fig.canvas.get_renderer()
        extent=self.zone_label.get_window_extent(renderer);scale=dpi/100
        width,height=extent.width,extent.height
        xmin,ymin=self.zone_screen.min(0);xmax,ymax=self.zone_screen.max(0)
        bounds=self.ax.bbox.extents
        lx=np.clip((xmin+xmax-width)/2,bounds[0]+10*scale,bounds[2]-width-10*scale)
        ly=ymax+18*scale
        if ly+height+8*scale>bounds[3]:ly=ymin-height-18*scale
        position=self.ax.transData.inverted().transform([lx,ly]);self.zone_label.set_position(position)
        self.zone_box=(lx-8*scale,ly-8*scale,lx+width+8*scale,ly+height+8*scale)
        assert not intersects(self.zone_box,(xmin,ymin,xmax,ymax)), 'Zone label must be outside polygon'
        # Replace the square symbol in the legend, leaving all axes/map extents intact.
        for text in self.fig.texts:
            if text.get_text().startswith('━━ Observed tracker'):
                text.set_text('━━ Observed tracker     ┄┄ Model forecast     Orange outline: current anomaly')
            if text.get_text().startswith('Up to 4 futures'):
                text.set_text('Up to 4 futures per anomalous object · endpoint % = rounded sample share · red = enters zone')
        self.fig.canvas.draw();self.background=self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.layout_receipt={}

    def _label(self,text,anchor,occupied):
        renderer=self.fig.canvas.get_renderer();extent=text.get_window_extent(renderer)
        box=place_label(anchor,(extent.width,extent.height),self.ax.bbox.extents,occupied,self.dpi/100)
        text.set_position(self.ax.transData.inverted().transform(box[:2]));text.set_ha('left')
        occupied.append((box[0]-4*self.dpi/100,box[1]-4*self.dpi/100,box[2]+4*self.dpi/100,box[3]+4*self.dpi/100))
        return box

    def frame(self,frame):
        data=self.data;row=data['timeline'][frame];scale=self.dpi/100
        self.fig.canvas.restore_region(self.background);self.clock.set_text(f"t = {row['time_s']:5.1f} s")
        for key,text in self.cards.items():text.set_text(cleanup_texts(row)[key])
        for i,(name,status,entry) in enumerate(self.object_rows):
            obj=row['objects'][i];name.set_text('Drone 01' if i==0 else f'Aircraft {i:02d}')
            state='Anomalous' if obj['anomalous'] else ('Normal' if obj['current_state_nll'] is not None else 'Pending')
            status.set_text(state);status.set_color(ORANGE if obj['anomalous'] else (GREEN if state=='Normal' else MUTED))
            evidence=obj['entry'];entry.set_text('—' if evidence is None else f"{100*evidence['entry_samples']/32:g}%")
        self.cards['scenario'].set_color(LEVEL_COLOURS[row['scenario']['status']]);self.entry_text.set_text('')
        active=data['object_groups'].get(frame,{})
        groups=[]
        for agent,frame_groups in active.items():
            groups.extend(dict(group,agent=agent,display_percent=percent) for group,percent in zip(frame_groups,integer_shares(frame_groups)))
        occupied=[self.zone_box];layout=[('zone',self.zone_box)]
        # Reserve all icons and endpoints before placing names and percentages.
        ends=[]
        for point in self.observed[:,frame]:
            x,y=self.ax.transData.transform(point);occupied.append((x-24*scale,y-24*scale,x+24*scale,y+24*scale))
        for group in groups:
            agent=group['agent'];origin=row['objects'][agent]['forecast_origin_frame'];prediction=data['forecasts'][origin]
            path=prediction['position'][group['representative_index'],agent,frame-origin:]
            end=self.ax.transData.transform(visual.enu_to_pixel(path[-1,:2],self.zoom));ends.append(end)
            x,y=end;occupied.append((x-7*scale,y-7*scale,x+7*scale,y+7*scale))
        for agent,(history,future,icon,endpoint,text,marker) in enumerate(self.tracks):
            obj=row['objects'][agent];point=self.observed[agent,frame]
            history.set_data(*self.observed[agent,:frame+1].T);icon.set_data([point[0]],[point[1]])
            delta=data['stream']['observed'][agent,frame,:2]-data['stream']['observed'][agent,max(0,frame-6),:2]
            icon.set_marker(marker.transformed(Affine2D().rotate(float(np.arctan2(delta[1],delta[0])))))
            icon.set_mec(ORANGE if obj['anomalous'] else BG)
            # Rotor gaps are narrower than the aircraft body: use a finer drone edge.
            icon.set_mew((.55 if agent==0 else .8) if obj['anomalous'] else 1)
            icon.set_path_effects([]) # Preserve the fine drone/aircraft details inside the orange edge.
            text.set_text('Drone 01' if agent==0 else f'Aircraft {agent:02d}');text.set_color(ORANGE if obj['anomalous'] else TEXT)
            box=self._label(text,self.ax.transData.transform(point),occupied);layout.append((f'object_{agent}',box))
            origin=obj['forecast_origin_frame'];prediction=data['forecasts'].get(origin)
            if prediction is not None and agent not in active:
                path=prediction['position'][prediction['medoid'],agent,frame-origin:]
                projected=visual.enu_to_pixel(path[:,:2],self.zoom);future.set_data(*np.vstack([point,projected]).T)
                endpoint.set_data([projected[-1,0]],[projected[-1,1]])
            else:future.set_data([],[]);endpoint.set_data([],[])
        for i,artists in enumerate(self.group_artists):
            line,endpoint,leader,label=artists
            for artist in artists:artist.set_visible(i<len(groups))
            if i>=len(groups):continue
            group=groups[i];agent=group['agent'];origin=row['objects'][agent]['forecast_origin_frame']
            path=data['forecasts'][origin]['position'][group['representative_index'],agent,frame-origin:]
            projected=visual.enu_to_pixel(path[:,:2],self.zoom);colour='#ff626b' if group['enters_zone'] else AMBER
            line.set_data(*np.vstack([self.observed[agent,frame],projected]).T)
            line.set_color(colour);line.set_linewidth(1.3+2*group['share']);line.set_alpha(.5+.5*group['share'])
            endpoint.set_data([projected[-1,0]],[projected[-1,1]]);endpoint.set_mfc(colour);endpoint.set_ms(6+4*group['share'])
            label.set_text(f"{group['display_percent']}%");label.set_color(colour)
            box=self._label(label,ends[i],occupied);layout.append((f'percentage_{i}',box))
            anchor=self.ax.transData.inverted().transform(box[:2]);leader.set_data([projected[-1,0],anchor[0]],[projected[-1,1],anchor[1]]);leader.set_color(colour)
        # Zone text and backing are always the last draw, above every path/node/text.
        for artist in [*self.artists,self.zone_label]:(artist.axes.draw_artist if artist.axes else self.fig.draw_artist)(artist)
        for i,(name,box) in enumerate(layout):
            assert not any(intersects(box,other) for _,other in layout[i+1:]),f'Overlapping labels at frame {frame}: {name}'
        actual=[]
        for name,text in [('zone',self.zone_label)]+[(f'object_{i}',t[4]) for i,t in enumerate(self.tracks)]+[(f'percentage_{i}',a[3]) for i,a in enumerate(self.group_artists) if a[3].get_visible()]:
            patch=text.get_bbox_patch();b=(patch or text).get_window_extent(self.fig.canvas.get_renderer())
            actual.append((name,tuple(b.extents)))
        for i,(name,box) in enumerate(actual):
            assert not any(intersects(box,other) for _,other in actual[i+1:]),f'Actual label collision: {frame} {name}'
        card_boxes=[]
        for i,items in enumerate(self.object_rows):
            for j,item in enumerate(items):
                bounds=item.get_window_extent(self.fig.canvas.get_renderer()).extents
                assert bounds[0]>=.735*self.fig.bbox.width and bounds[2]<=.98*self.fig.bbox.width, 'Object row overflow'
                assert bounds[1]>=.045*self.fig.bbox.height and bounds[3]<=.267*self.fig.bbox.height, 'Object row overflow'
                card_boxes.append(tuple(bounds))
        for i,box in enumerate(card_boxes):assert not any(intersects(box,other) for other in card_boxes[i+1:]), 'Object cells overlap'
        self.layout_receipt[frame]=dict(object_card_readable=True,object_card_fontsize=13,time_s=row['time_s'],labels=[dict(name=name,bounds=list(box)) for name,box in actual],
            zone_label_outside=True,labels_do_not_overlap=True)
        return Image.fromarray(np.asarray(self.fig.canvas.buffer_rgba())[...,:3].copy())
