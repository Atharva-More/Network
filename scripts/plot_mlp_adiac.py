"""Reconstruct Figure 1 from MLP.zip using recorded experiment results."""
import io, re, sys, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

source = Path(sys.argv[1] if len(sys.argv) > 1 else '/mnt/data/MLP.zip')
with zipfile.ZipFile(source) as z:
    def csv(suffix):
        matches = [n for n in z.namelist() if n.endswith(suffix)]
        assert len(matches) == 1, (suffix, matches)
        return pd.read_csv(io.BytesIO(z.read(matches[0])))

    audit = csv('UCR_MLP_Experiment_Summary_2026-11-12.csv')
    a1 = csv('A1_official_test_results.csv')
    o5 = csv('O5_summary.csv')
    selections = [
        audit.loc[audit.experiment == 'E1','selected_test_accuracy_pct'].to_numpy(float),
        audit.loc[audit.experiment == 'E3','selected_test_accuracy_pct'].to_numpy(float),
        a1.test_accuracy.to_numpy(float)*100,
        o5.selected_test_accuracy_history.to_numpy(float)*100,
    ]
    historical = []
    patt = re.compile(r'^\s*390/390 .* - loss: [0-9.eE+-]+ - acc: [0-9.eE+-]+ - val_loss: [0-9.eE+-]+ - val_acc: ([0-9.eE+-]+)')
    for name in sorted(n for n in z.namelist() if re.search(r'keras2017_seed_\d+\.log$',n)):
        values = [float(m.group(1))*100 for line in z.read(name).decode('utf8','ignore').splitlines() if (m:=patt.match(line))]
        assert len(values)==5000, (name,len(values))
        historical.append(values)
    o5_curves = []
    for name in sorted(n for n in z.namelist() if re.search(r'/O5/seed_\d+_history\.csv$',n)):
        history = pd.read_csv(io.BytesIO(z.read(name)))
        assert len(history)==5000
        o5_curves.append(history.val_accuracy.to_numpy(float)*100)

assert [len(x) for x in selections]==[1,5,5,5]
assert len(historical)==5 and len(o5_curves)==5
means=np.array([np.mean(x) for x in selections])
historical_mean=np.mean(historical,axis=0)
historical_smooth=pd.Series(historical_mean).rolling(50,min_periods=1,center=True).mean().to_numpy()
o5_mean=np.mean(o5_curves,axis=0)
sample_epochs=np.arange(250,5001,250)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'svg.fonttype':'none'})
fig,(left,right)=plt.subplots(1,2,figsize=(11.5,4.45))
labels=['Original code\nTF 1.15.5','Original code\nKeras 1.2.2','Compact A1\n186 epochs','Compact O5\n5000 epochs']
positions=np.arange(4)
left.bar(positions,means,width=.67,edgecolor='black',linewidth=.5,alpha=.76)
for i,values in enumerate(selections):
    shifts=np.linspace(-.15,.15,len(values)) if len(values)>1 else [0]
    left.scatter(i+np.asarray(shifts),values,s=26,zorder=5,edgecolors='black',linewidths=.4)
    left.annotate(f'{means[i]:.1f}',(i,means[i]),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9,weight='bold')
left.axhline(75.2,linestyle='--',linewidth=1.35,color='.25')
left.text(.03,77.1,'Paper: 75.2%',fontsize=8)
left.set_xticks(positions,labels)
left.set_ylim(0,87)
left.set_yticks(range(0,81,10))
left.set_ylabel('Test accuracy (%)')
left.set_title('(a) Test accuracy by setting')
left.grid(axis='y',alpha=.18)
left.set_axisbelow(True)

right.plot(np.arange(1,5001),historical_smooth,linewidth=1.4,label='Published MLP (2017), 50-epoch smoothing')
right.plot(sample_epochs,o5_mean[sample_epochs-1],marker='o',markersize=3.8,linewidth=1.4,label='Compact MLP (O5), every 250 epochs')
right.set_xlim(0,5000)
right.set_ylim(0,80)
right.set_xticks(range(0,5001,1000))
right.set_xlabel('Epoch')
right.set_ylabel('Test accuracy (%)')
right.set_title('(b) Test accuracy during training')
right.legend(loc='lower right',fontsize=8)
right.grid(alpha=.18)
right.set_axisbelow(True)
fig.subplots_adjust(left=.067,right=.985,bottom=.16,top=.9,wspace=.25)
output=Path(__file__).resolve().parent
fig.savefig(output/'mlp_on_adiac_matplotlib.png',dpi=350)
fig.savefig(output/'mlp_on_adiac_matplotlib.svg')
plt.close(fig)
print('Mean test accuracy (%)',means)
print('Created PNG and SVG.')
