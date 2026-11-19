"""Display-only causal icon filtering and age-coloured raw tracker trails."""
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from src.eval import demo_gif as visual
from src.integrated.visual_cleanup import CleanupRenderer
from src.integrated.presentation import CYAN,GREEN


def causal_positions(observed,window=7):
    """Fit a line to at most 2s of past samples, evaluate at the current time.

This reduces noise without a moving average's lag on constant-speed motion.
It is solely a display transform, not a replacement tracker or a model input.
"""
    observed=np.asarray(observed,dtype=float);result=observed.copy()
    for frame in range(observed.shape[1]):
        start=max(0,frame-window+1);segment=observed[:,start:frame+1]
        t=np.arange(segment.shape[1],dtype=float);t-=t.mean()
        if len(t)<3:continue
        slope=np.einsum('t,atd->ad',t,segment)/(t@t)
        result[:,frame]=segment.mean(axis=1)+slope*t[-1]
    return result


def predicted_headings(data,display_position,tau_s=.6):
    """Causal near-term joint-medoid tangent, circularly eased for display.

    Use only the latest already-issued forecast, over its next ~2s. Before the
    first forecast (6s), fall back to the filtered observed displacement.
    An explicit playback start uses its preceding6s history for the initial
    visible heading. NaN only represents direction genuinely absent from inputs.
    """
    n,t,_=display_position.shape;result=np.full((n,t),np.nan)
    alpha=1-np.exp(-1/(3*tau_s))
    playback_start=int(data.get('playback_start_frame',0))
    for frame,row in enumerate(data['timeline']):
        for agent,obj in enumerate(row['objects']):
            # Initialize the first visible pose from its supplied past history,
            # not from an arbitrary angle or a future point in the playback.
            if frame==playback_start and playback_start>0:
                history=display_position[agent,frame,:2]-display_position[agent,max(0,frame-18),:2]
                if np.isfinite(history).all() and np.linalg.norm(history)>1e-6:
                    result[agent,frame]=np.arctan2(history[1],history[0])
                    continue
            origin=obj['forecast_origin_frame'];forecast=data['forecasts'].get(origin)
            if forecast is not None:
                path=forecast['position'][forecast['medoid'],agent]
                start=max(0,frame-origin);end=min(start+6,len(path)-1)
                delta=path[end,:2]-path[start,:2]
            else:delta=display_position[agent,frame,:2]-display_position[agent,max(0,frame-6),:2]
            previous=result[agent,frame-1] if frame else np.nan
            if not np.isfinite(delta).all() or np.linalg.norm(delta)<=1e-6:
                result[agent,frame]=previous
                continue
            target=np.arctan2(delta[1],delta[0])
            # Unknown is not east: initialize from the first causal direction.
            result[agent,frame]=target if not np.isfinite(previous) else previous+alpha*np.arctan2(np.sin(target-previous),np.cos(target-previous))
    return result


def trail_alphas(frame,hz=3,seconds=10):
    start=max(0,frame-round(seconds*hz));ends=np.arange(start+1,frame+1)
    return start,.12+.83*np.clip(1-(frame-ends)/(seconds*hz),0,1)**1.3


class TrackStyleRenderer(CleanupRenderer):
    def __init__(self,data,dpi=100):
        super().__init__(data,dpi=dpi)
        self.filtered=causal_positions(data['stream']['observed'])
        self.icon_pixels=visual.enu_to_pixel(self.filtered[...,:2],self.zoom)
        self.icon_headings=predicted_headings(data,self.filtered)
        self.trails=[]
        for i in range(len(self.tracks)):
            colour=CYAN if i in self.drone_agents else GREEN
            trail=LineCollection([],linewidths=3,animated=True,capstyle='round',zorder=3)
            self.ax.add_collection(trail);self.trails.append((trail,colour))
            # Draw before icons, labels and forecast paths.
            self.artists.insert(0,trail)
        for text in self.fig.texts:
            if text.get_text().startswith('━━ Observed tracker'):
                text.set_text('━━ Raw tracker history · recent 10 s fades with age     ┄┄ Model forecast')
            elif text.get_text().startswith('Up to 4 futures'):
                text.set_text('Display-smoothed icons · forecast-based heading · endpoint % = rounded sample share · red = enters zone')
        self.fig.canvas.draw();self.background=self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def display_point(self,agent,frame):return self.icon_pixels[agent,frame]

    def display_heading(self,agent,frame):return float(self.icon_headings[agent,frame])

    def draw_history(self,history,agent,frame):
        super().draw_history(history,agent,frame)
        history.set_alpha(.16);history.set_linewidth(1.2);history.set_path_effects([])
        start,alphas=trail_alphas(frame);points=self.observed[agent,start:frame+1]
        trail,colour=self.trails[agent]
        trail.set_segments(np.stack([points[:-1],points[1:]],axis=1))
        colours=np.tile(to_rgba(colour),(len(alphas),1));colours[:,3]=alphas
        trail.set_color(colours)
