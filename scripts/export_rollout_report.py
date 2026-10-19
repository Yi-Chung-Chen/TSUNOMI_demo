"""Report figures from the six existing seeded Demo cases; no model execution."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from src.eval.load_rollout import load_rollout

INK='#19354b'
BLUE='#338bd0'
ORANGE='#e08724'
PHASE={'climbing':'Climb','descending':'Descent','level':'Level flight','ground':'Ground movement'}


def draw(ax,data,agent,height=False):
    gt=data['gt_position'][agent].astype(float).copy()
    gt[~data['valid_mask'][agent]]=np.nan
    pred=data['position'][:,agent].astype(float)
    assert pred.shape==(32,90,3)
    t=np.arange(1,91)/3
    if height:
        for rollout in pred:
            ax.plot(np.r_[0,t],np.r_[gt[18,2],rollout[:,2]],color=BLUE,alpha=.23,lw=1.)
        ax.plot(np.arange(-18,1)/3,gt[:19,2],color=INK,lw=2.6)
        ax.plot(np.r_[0,t],gt[18:,2],color=ORANGE,lw=2.2)
        ax.scatter(0,gt[18,2],s=45,color=INK,marker='D',zorder=5,edgecolors='white',linewidths=.7)
        ax.axvline(0,color=INK,alpha=.25,ls='--',lw=.8)
        ax.set(xlabel='Time from forecast start (s)',ylabel='Height above reference ground (m)',xlim=(-6,30))
    else:
        for rollout in pred:
            trajectory=np.vstack((gt[18],rollout))
            ax.plot(trajectory[:,0]/1000,trajectory[:,1]/1000,color=BLUE,alpha=.23,lw=1.)
        ax.plot(gt[:19,0]/1000,gt[:19,1]/1000,color=INK,lw=2.6)
        ax.plot(gt[18:,0]/1000,gt[18:,1]/1000,color=ORANGE,lw=2.2)
        ax.scatter(*gt[18,:2]/1000,s=45,color=INK,marker='D',zorder=5,edgecolors='white',linewidths=.7)
        ax.set(xlabel='East from reference origin (km)',ylabel='North from reference origin (km)',aspect='equal')
        ax.margins(.12)
        # Preserve equal-distance geometry without collapsing straight tracks
        # into unreadable strips; expand limits without trimming any sample.
        xlim,ylim=ax.get_xlim(),ax.get_ylim()
        center=np.array([np.mean(xlim),np.mean(ylim)])
        width=max(xlim[1]-xlim[0],(ylim[1]-ylim[0])/.65)
        ax.set(xlim=(center[0]-width/2,center[0]+width/2),
            ylim=(center[1]-.65*width/2,center[1]+.65*width/2))
    ax.grid(color='#dfe7ee',lw=.7,alpha=.8)
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    for spine in ['bottom','left']:ax.spines[spine].set_color('#bac8d3')
    ax.tick_params(colors='#526879',labelsize=9)


def legend(fig,y):
    handles=[Line2D([],[],color=INK,lw=2.6,label='Observed history'),
        Line2D([],[],color=ORANGE,lw=2.2,label='Recorded future'),
        Line2D([],[],color=BLUE,lw=1.5,alpha=.7,label='Simulated futures (32)'),
        Line2D([],[],color=INK,marker='D',ls='',label='Forecast start')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,y),ncol=4,frameon=False,fontsize=10)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args();root=args.root.resolve();out=root/'outputs/quarterly_rollouts';out.mkdir(parents=True,exist_ok=True)
    selection=root/'outputs/demo/selection.json';selected=json.loads(selection.read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':INK,
        'axes.labelcolor':INK,'axes.titlecolor':INK,'pdf.fonttype':42,'savefig.facecolor':'white'})
    overview,overview_axes=plt.subplots(2,3,figsize=(15,10))
    overview.subplots_adjust(top=.79,bottom=.11,hspace=.43,wspace=.32)
    overview.suptitle('Aircraft trajectory simulation',x=.07,y=.97,ha='left',fontsize=23,fontweight='bold')
    overview.text(.07,.905,'6 s observed history  /  30 s forecast  /  32 sampled futures per scene',fontsize=12,color='#526879')
    legend(overview,.865)
    records=[]
    for index,row in enumerate(selected['windows'],1):
        source=root/row['path'];before=hashlib.sha256(source.read_bytes()).hexdigest()
        data=load_rollout(source);agent=row['agent'];phase=PHASE[row['kind']]
        fig,axes=plt.subplots(1,2,figsize=(13,5.8),gridspec_kw={'width_ratios':[1.1,1]})
        fig.subplots_adjust(top=.67,bottom=.18,left=.08,right=.97,wspace=.32)
        fig.suptitle('Aircraft trajectory simulation',x=.06,y=.97,ha='left',fontsize=22,fontweight='bold')
        fig.text(.06,.865,f'Case {index:02d} · {phase}   |   6 s history → 30 s forecast   |   32 sampled futures',fontsize=11,color='#526879')
        legend(fig,.815)
        draw(axes[0],data,agent);draw(axes[1],data,agent,height=True)
        axes[0].set_title('Horizontal trajectory',loc='left',fontsize=12,fontweight='bold',pad=12)
        axes[1].set_title('Vertical profile',loc='left',fontsize=12,fontweight='bold',pad=12)
        fig.text(.06,.055,f'SYGARD · O’Hare aircraft data · {row["date"]}',fontsize=9,color='#526879')
        stem=f'case_{index:02d}_{row["kind"]}'
        for suffix in ['png','pdf']:fig.savefig(out/f'{stem}.{suffix}',dpi=300)
        plt.close(fig)
        ax=overview_axes.flat[index-1];draw(ax,data,agent)
        ax.set_title(f'{index:02d}  {phase}',loc='left',fontsize=12,fontweight='bold',pad=10)
        assert hashlib.sha256(source.read_bytes()).hexdigest()==before
        records.append(dict(**row,input_sha256=before,png=f'{stem}.png',pdf=f'{stem}.pdf',rollouts=32))
    overview.text(.07,.025,'SYGARD · O’Hare aircraft data | Six illustrative validation scenes',fontsize=10,color='#526879')
    for suffix in ['png','pdf']:overview.savefig(out/f'rollout_overview.{suffix}',dpi=300)
    plt.close(overview)
    caption=('Qualitative aircraft trajectory rollouts on SYGARD O’Hare data. Given 6 s of observed history (dark blue), '
        'the behaviour-cloning model generates 32 sampled futures over a 30 s horizon (light blue). '
        'The recorded future is shown in orange; diamonds mark forecast initialization. '
        'Individual figures show horizontal trajectories and vertical profiles for a selected aircraft from each joint scene. '
        'The same six seeded, flight-phase-stratified cases from the existing Demo are shown, without ranking prediction errors. '
        'These illustrative validation examples are not independent test evidence or calibrated prediction intervals.')
    manifest=dict(checkpoint='epoch_016.ckpt',model_execution=False,training=False,history_s=6,future_s=30,
        rollout_count=32,sampling='topk_prob K=5, temperature=1.0',selection=selected['rule'],seed=selected['seed'],
        selection_sha256=hashlib.sha256(selection.read_bytes()).hexdigest(),figures=records,caption=caption)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'CAPTION.md').write_text('# Suggested report caption\n\n'+caption+'\n\n'
        'PNG: 300 dpi (individual3900×1740; overview4500×3000). PDF: vector figures. '
        'All32 rollout samples are displayed, without best-of selection or clipping. '
        'Height is relative to the fixed166m reference ground, not a terrain-derived AGL measure. '
        'Checkpoint and dictionary selection used10-03 validation data; these examples include10-03/10-04.\n')
    with ZipFile(out/'quarterly_rollout_figures.zip','w',compression=ZIP_DEFLATED) as archive:
        for path in sorted(out.iterdir()):
            if path.suffix in ['.png','.pdf','.md','.json']:archive.write(path,arcname=path.name)
    print(json.dumps(dict(output=str(out),individual_figures=len(records),overview=True,model_execution=False),indent=2))


if __name__=='__main__':main()
