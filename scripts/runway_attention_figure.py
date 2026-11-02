"""Observed-history-only CPU attention diagnostic using cached map tiles only."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data.dataset import AirDataset
from src.model.air_smart import AirSMART
from src.common.units import to_metres
from src.eval import demo_gif as visual
from src.integrated.presentation import cached_basemap
from scripts.evaluate_runway_model import read_checkpoint


def main():
    root=Path(__file__).resolve().parents[1];output=root/'outputs/runway_anchor';torch.set_num_threads(1)
    saved=torch.load(read_checkpoint(root,'anchor'),map_location='cpu',weights_only=False)
    model=AirSMART(**saved['hyper_parameters']);model.load_state_dict(saved['state_dict'],strict=True);model.eval()
    dataset=AirDataset(root/'data/processed',root/'data/splits','val');indices=[]
    for i,row in enumerate(dataset.windows):
        if row['scenario'] not in getattr(main,'dates',{}):
            if not hasattr(main,'dates'):main.dates={}
            with np.load(root/'data/processed'/f"{row['scenario']}.npz") as a:main.dates[row['scenario']]=str(a['date'])
        if main.dates[row['scenario']]=='2026-10-04':indices.append(i)
    selected=np.random.default_rng(817).choice(indices,3,replace=False).tolist()
    decoder=model.decoder.agent_decoder;anchors=to_metres(decoder.anchor_position,model.token_processor.meta['pos_scale']).cpu().numpy()
    with (root/'assets/ord_runway_ends.csv').open() as f:idents=[r['ident'] for r in csv.DictReader(f)]
    panels=[]
    for index in selected:
        graph=dataset[index];history=graph.clone()
        for key in ['position','heading','valid_mask','on_ground','alt_fallback']:
            history['agent'][key]=history['agent'][key][:,:19]
        with torch.no_grad():_,tokens=model(history)
        edge,features=decoder.anchor_edges(tokens['gt_pos'].float(),tokens['gt_heading'].float(),tokens['valid_mask'])
        mean=torch.stack([layer.attention_weight/layer.num_heads for layer in decoder.anchor_layers]).mean(0)
        steps=tokens['valid_mask'].shape[1];weights=[]
        for agent in range(graph['agent'].num_nodes):
            take=edge[1]==agent*steps+model.current_node
            assert int(take.sum())==16
            values=mean[take].detach().numpy();assert abs(values.sum()-1)<1e-5
            weights.append(values)
        current=to_metres(history['agent']['position'][:,-1],model.token_processor.meta['pos_scale']).numpy()
        past=to_metres(history['agent']['position'],model.token_processor.meta['pos_scale']).numpy()
        panels.append(dict(scenario=graph['scenario'],start_frame=graph['start_frame'],agent_ids=graph['agent']['id'],
            current=current,history=past,weights=np.array(weights)))
    bounds=visual.view_bounds(np.concatenate([anchors[:,:2],*[p['current'][:,:2] for p in panels]]),aspect=1.15)
    image,zoom,pb=cached_basemap(bounds,root/'data/basemap_cache')
    fig,axes=plt.subplots(1,3,figsize=(18,6),facecolor='#0b1622')
    ap=visual.enu_to_pixel(anchors[:,:2],zoom)
    for ax,panel in zip(axes,panels):
        ax.imshow(image,extent=(pb[0,0],pb[1,0],pb[1,1],pb[0,1]));ax.set(xlim=(pb[0,0],pb[1,0]),ylim=(pb[1,1],pb[0,1]),aspect='equal');ax.axis('off')
        ax.scatter(*ap.T,s=18,c='#c77dff',zorder=4)
        for point,ident in zip(ap,idents):ax.annotate(ident,point,xytext=(3,3),textcoords='offset points',fontsize=6,color='#e4c7ff')
        positions=visual.enu_to_pixel(panel['current'][:,:2],zoom)
        histories=visual.enu_to_pixel(panel['history'][...,:2],zoom)
        for a,(point,weight) in enumerate(zip(positions,panel['weights'])):
            colour=visual.COLOURS[a%len(visual.COLOURS)] if hasattr(visual,'COLOURS') else ['#4cc9f0','#80ed99','#ffd166','#ff626b'][a%4]
            for anchor,value in zip(ap,weight):ax.plot([point[0],anchor[0]],[point[1],anchor[1]],lw=.3+12*float(value),color=colour,alpha=.65)
            ax.plot(*histories[a].T,lw=1,color=colour);ax.scatter(*point,s=30,color=colour,zorder=5)
            ax.annotate(str(panel['agent_ids'][a]),point,xytext=(4,-9),textcoords='offset points',fontsize=7,color=colour)
        ax.set_title(f"{panel['scenario']} · t={panel['start_frame']/3+6:.1f}s",color='#e8eef5',fontsize=10)
    fig.suptitle('Runway-anchor attention · mean over 4 heads and 4 layers',color='#e8eef5',fontsize=15)
    fig.text(.5,.025,'Line width ∝ attention weight · observed 6 s history only · fixed seed817 · cached OSM tiles · diagnostic, not causal explanation',ha='center',color='#8fa3b8',fontsize=10)
    fig.text(.99,.005,'© OpenStreetMap contributors',ha='right',color='#8fa3b8',fontsize=8)
    fig.savefig(output/'anchor_attention.png',dpi=180,facecolor=fig.get_facecolor());plt.close(fig)
    serial=[]
    for panel in panels:serial.append({k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in panel.items()})
    (output/'attention_weights.json').write_text(json.dumps(dict(seed=817,dataset_indices=selected,anchor_idents=idents,
        average='all4heads/all4layers',history_only=True,map_zoom=zoom,network_requests=False,windows=serial),indent=2)+'\n')
    print('Attention diagnostic complete',flush=True)

if __name__=='__main__':main()
