import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams.update({
                "text.usetex": True,
                "font.family": "sans-serif",
                "font.sans-serif": ["Helvetica"]})

for d in ['sat', 'u2', 'u4', 'google', 'bats']:
    lm = pd.read_csv('./experiments_results/summary/main2.test.prediction.{}.csv'.format(d))
    we = pd.read_csv('./experiments_results/summary/statistics.test.prediction.{}.csv'.format(d))
    full = pd.concat([lm, we])
    full['model'] = ['RoBERTa'] * len(lm) + ['FastText']*len(we)
    full['accuracy'] = full['prediction'] == full['answer']
    if d == 'sat':
        full['prefix'] = full['prefix'].apply(lambda x: 'SAT' if 'FROM REAL SAT' else 'not SAT')
    elif d == 'bats':
        full['prefix'] = full['prefix'].apply(lambda x: x.split(' [')[-1].split(']')[0].replace('_', ' '))

    g = full.groupby(['prefix', 'model']).accuracy.mean()
    f = g.to_frame()
    f.reset_index(inplace=True)
    # plt.grid()
    ax = sns.barplot(x='prefix', y='accuracy', hue='model', data=f)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels)
    ax.set_xlabel(None)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.tick_params(labelsize=12)
    # plt.xticks(rotation=60)
    fig = ax.get_figure()
    plt.tight_layout()
    fig.savefig('./experiments_results/summary/main2_figure/bar.{}.png'.format(d))
    plt.close()
