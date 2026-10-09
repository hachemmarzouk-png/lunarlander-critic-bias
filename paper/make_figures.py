"""Rebuild the article figures/tables directly from the archived experimental CSV logs.

This script never simulates or fabricates observations.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from rlbias.aggregate import load_results, final_by_seed, permutation_pvalue
OUT=Path(__file__).resolve().parent
blue='#285a82'; teal='#247d78'; ochre='#c47a32'; red='#b95445'
colors={('ddpg','no'):blue, ('ddpg','yes'):red, ('td3','no'):teal, ('td3','yes'):ochre}
labels={('ddpg','no'):'DDPG',('ddpg','yes'):'DDPG + LN',('td3','no'):'TD3',('td3','yes'):'TD3 + LN'}
order=list(labels)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.0, 'axes.titlesize':9.6,
    'axes.labelsize':9.0,'xtick.labelsize':8.2,'ytick.labelsize':8.2,
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.edgecolor':'#a6b0b9','axes.linewidth':.65,'pdf.fonttype':42,'savefig.transparent':True})
all_df=load_results(BASE/'results')
all_final=final_by_seed(all_df,3)
assert len(all_final)==30, f"Expected 30 trained runs, found {len(all_final)}"
assert all(len(all_final[(all_final.algorithm==a)&(all_final.layer_norm==l)]) == 5
           for a,l in order + [('ddpg','actor'),('ddpg','critic')]), 'Incomplete 5-seed design'
df=all_df[all_df.layer_norm.isin(['no','yes'])].copy()
final=all_final[all_final.layer_norm.isin(['no','yes'])].copy()
assert len(final)==20

def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.065)
    fig.savefig(OUT/(name+'.png'),dpi=200,bbox_inches='tight',pad_inches=.065)
    plt.close(fig)

# Fig 1: readable two-panel time series, point estimates and 5-seed SD.
fig,axs=plt.subplots(1,2,figsize=(7.25,3.35),gridspec_kw={'wspace':.35})
for ax,metric in zip(axs,['mean_return','q1_bias']):
    for cfg in order:
        part=df[(df.algorithm==cfg[0])&(df.layer_norm==cfg[1])&(df.step>0)]
        stat=part.groupby('step')[metric].agg(['mean','std']).reset_index()
        x=stat.step.to_numpy()/1000; y=stat['mean'].to_numpy(); sd=stat['std'].to_numpy()
        ax.plot(x,y,linewidth=1.9,color=colors[cfg],ls='--' if cfg[1]=='yes' else '-',
                marker='s' if cfg[1]=='yes' else 'o', markersize=2.9,markevery=2,
                label=labels[cfg])
        ax.fill_between(x,y-sd,y+sd,color=colors[cfg],alpha=.085,lw=0)
    ax.set_xlim(25,300);ax.set_xticks([50,100,150,200,250,300]);ax.grid(alpha=.16,lw=.5)
    ax.set_xlabel('Interactions avec l’environnement (× 1 000)')
axs[0].set_title('A · Performance de la politique',loc='left',fontweight='bold')
axs[0].set_ylabel('Retour par épisode (non actualisé)')
axs[0].set_ylim(-230,340)
axs[0].axhline(200,ls=':',color='#616b75',lw=1)
# 200-point guide is explained in caption to avoid label/curve overlap
axs[1].set_title('B · Biais signé du critic',loc='left',fontweight='bold')
axs[1].set_ylabel('Moyenne de Q₁(s,a) − G (points)')
axs[1].set_yscale('symlog',linthresh=45,linscale=1.0)
axs[1].set_ylim(-85,4500); axs[1].set_yticks([-50,0,50,150,500,1500]);axs[1].set_yticklabels(['−50','0','50','150','500','1 500'])
axs[1].axhline(0,color='#63727c',lw=.9)
handles=[Line2D([],[],color=colors[k],ls='--' if k[1]=='yes' else '-',lw=1.9,
                marker='s' if k[1]=='yes' else 'o',ms=4,label=labels[k]) for k in order]
fig.legend(handles=handles,ncol=4,loc='lower center',bbox_to_anchor=(.5,-.135),frameon=False,
           fontsize=9,handlelength=2.4,columnspacing=1.4)
save(fig,'figure_evolution')

# Fig 2: variation between seeds + within-TD3 comparison of the two critics.
fig, axs=plt.subplots(1,2,figsize=(7.22,2.95),gridspec_kw={'wspace':.42})
ax=axs[0]
for cfg in order:
    part=final[(final.algorithm==cfg[0])&(final.layer_norm==cfg[1])].sort_values('seed')
    ax.scatter(part.q1_bias,part.mean_return,s=33,c=colors[cfg],alpha=.9,
               marker='s' if cfg[1]=='yes' else 'o',edgecolor='white',lw=.55)
ax.set_xscale('symlog',linthresh=35)
ax.set_xticks([-25,0,100,1000]);ax.set_xticklabels(['−25','0','100','1 000'])
ax.axvline(0,color='#63727c',lw=.9)
ax.set_xlabel('Biais final moyen (points)');ax.set_ylabel('Retour final moyen')
ax.set_title('A · 20 entraînements, un point par graine',loc='left',fontweight='bold',fontsize=9)
ax.grid(alpha=.16,lw=.5)
ax=axs[1]
for i,ln in enumerate(['no','yes']):
    part=final[(final.algorithm=='td3')&(final.layer_norm==ln)].sort_values('seed')
    # connected points per seed, horizontal spacing for visibility
    for j,(_,r) in enumerate(part.iterrows()):
        y=j+(0 if ln=='no' else 6)
        ax.plot([r.q1_bias,r.qmin_bias],[y,y],color=colors[('td3',ln)],lw=1.15,alpha=.85)
        ax.scatter([r.q1_bias],[y],c=[colors[('td3',ln)]],s=20,marker='o',zorder=3)
        ax.scatter([r.qmin_bias],[y],facecolors='white',edgecolors=[colors[('td3',ln)]],s=28,marker='s',zorder=3,lw=1.15)
ax.axvline(0,color='#63727c',lw=.9)
ax.set_yticks([2,8]);ax.set_yticklabels(['TD3','TD3 + LN'])
ax.set_xlabel('Biais final (points)');ax.set_xlim(-30,2)
ax.set_title('B · TD3 : Q₁ versus min(Q₁,Q₂)',loc='left',fontweight='bold',fontsize=9)
ax.grid(axis='x',alpha=.17,lw=.5)
save(fig,'figure_details')

# Light-weight tables: sample dispersion rather than overprecise inferential p-values.
def fmt(n,dec=0):
    t=f'{n:+.{dec}f}' if n<0 else f'{n:.{dec}f}'
    t=t.replace('-','−').replace('+','+')
    return t

def latex_num(n,d=0,sign=False):
    s=f'{n:+,.{d}f}' if sign else f'{n:,.{d}f}'
    return s.replace(',','\\,').replace('.', '{,}')
rows=[]
for k in order:
    part=final[(final.algorithm==k[0])&(final.layer_norm==k[1])]
    data=[]
    for col in ['mean_return','q1_bias','q1_mae']:
        mean,sd=part[col].mean(),part[col].std(ddof=1)
        data.append(f'${latex_num(mean,0,col=="q1_bias")} \\pm {latex_num(sd,0)}$')
    rows.append((labels[k],*data))
lines=[r'\begin{tabular}{@{}lrrr@{}}',r'\toprule',r'\textbf{Configuration} & \textbf{Retour} & \textbf{Biais $Q_1-G$} & \textbf{Erreur $|Q_1-G|$} \\',r'\midrule']
for name,a,b,c in rows: lines.append(f'{name} & {a} & {b} & {c} \\\\')
lines += [r'\bottomrule',r'\end{tabular}']
(OUT/'table_results_revised.tex').write_text('\n'.join(lines))

# Compare the two groups separately for each outcome: one p-value per table row.
contrasts = [
    ('DDPG', 'TD3', ('ddpg', 'no'), ('td3', 'no')),
    ('DDPG', 'DDPG+LN', ('ddpg', 'no'), ('ddpg', 'yes')),
    ('TD3', 'TD3+LN', ('td3', 'no'), ('td3', 'yes')),
]
lines = [r'\begin{tabular}{@{}llrr@{}}', r'\toprule',
         r'\textbf{Comparaison (avant $\to$ après)} & \textbf{Mesure} & \textbf{Différence} & $\boldsymbol p$ \\',
         r'\midrule']
for before_name, after_name, before, after in contrasts:
    for index, (metric, outcome) in enumerate([('mean_return', 'Retour'), ('q1_bias', 'Biais')]):
        first = final[(final.algorithm == before[0]) & (final.layer_norm == before[1])][metric].to_numpy(float)
        second = final[(final.algorithm == after[0]) & (final.layer_norm == after[1])][metric].to_numpy(float)
        delta = second.mean() - first.mean()
        p_value = permutation_pvalue(first, second)
        comparison = f'{before_name} $\\to$ {after_name}' if index == 0 else ''
        lines.append(f'{comparison} & {outcome} & ${latex_num(delta, 0, True)}$ & '
                     + '$' + f'{p_value:.3f}'.replace('.', r'{,}') + r'$ \\')
lines += [r'\bottomrule', r'\end{tabular}']
(OUT / 'table_contrasts_revised.tex').write_text('\n'.join(lines))
# The 10 additional runs are kept separate from the primary 2x2 design.
ablation_order=[('no','Sans LN'),('actor','LN actor seul'),('critic','LN critic seul'),('yes','LN actor + critic')]
ablation=all_final[all_final.algorithm=='ddpg'].copy()
lines=[r'\begin{tabular}{@{}lrr@{}}',r'\toprule',
       r'\textbf{DDPG} & \textbf{Retour} & \textbf{Biais $Q_1-G$} \\',r'\midrule']
for ln,name in ablation_order:
    group=ablation[ablation.layer_norm==ln]
    vals=[]
    for col in ['mean_return','q1_bias']:
        mu=float(group[col].mean()); sd=float(group[col].std(ddof=1))
        vals.append(f'${latex_num(mu,0,col=="q1_bias")} \\pm {latex_num(sd,0)}$')
    lines.append(f'{name} & '+ ' & '.join(vals)+r' \\')
lines += [r'\bottomrule',r'\end{tabular}']
(OUT/'table_ablation_revised.tex').write_text('\n'.join(lines))
print('All 30 runs loaded; article figures and tables regenerated.')
