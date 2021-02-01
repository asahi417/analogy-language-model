import seaborn as sns
import matplotlib.pyplot as plt
import alm
import os
import json
from itertools import product

plt.rcParams.update({"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]})

os.makedirs('./experiments_results/summary/main2_figure', exist_ok=True)
export_prefix = 'main2'
df = alm.get_report(export_prefix=export_prefix)
# g = df.groupby(['data'])
# mean_acc = json.loads(g.accuracy.mean().to_json())
# df['accuracy_mean'] = df['data'].apply(lambda x: mean_acc[x])
# df['accuracy'] = df['accuracy'] - df['accuracy_mean']
df['accuracy'] = df['accuracy'] * 100
df['accuracy'] = df['accuracy'].round(1)
data = ['sat', 'u2', 'u4', 'google', 'bats']
model = ['roberta-large', 'gpt2-xl']
big_group = df.groupby(['data', 'model', 'ppl_pmi_alpha', 'negative_permutation_weight']).accuracy.max()

sns.set_theme(style="darkgrid")


def plot(d, m, accuracy):
    fig = plt.figure()
    fig.clear()
    print(accuracy)
    accuracy = accuracy - accuracy[0.0][0.0]
    accuracy = accuracy.to_frame()
    accuracy.reset_index(inplace=True)
    accuracy = accuracy.pivot(index='ppl_pmi_alpha', columns='negative_permutation_weight', values='accuracy')
    accuracy = accuracy.reindex([0.4, 0.2, 0.0, -0.2, -0.4])
    sns_plot = sns.heatmap(accuracy, annot=True, fmt="g", cbar=False, annot_kws={"fontsize": 15})
    sns_plot.set_xlabel(r'$\beta$', fontsize=18)
    sns_plot.set_ylabel(r'$\alpha$', fontsize=18)
    sns_plot.tick_params(labelsize=15)
    fig = sns_plot.get_figure()
    plt.tight_layout()
    fig.savefig('./experiments_results/summary/main2_figure/heatmap.alpha_beta.{}.{}.png'.format(d, m))
    plt.close()


# plot('all', 'all', df.groupby(['ppl_pmi_alpha', 'negative_permutation_weight']).accuracy.max())
plot('all', 'all', df.groupby(['ppl_pmi_alpha', 'negative_permutation_weight']).accuracy.mean())
for m_ in model:
    plot('all', m_, df.groupby(['model', 'ppl_pmi_alpha', 'negative_permutation_weight']).accuracy.mean()[m_])

for d_ in data:
    plot(d_, 'all', df.groupby(['data', 'ppl_pmi_alpha', 'negative_permutation_weight']).accuracy.mean()[d_])

for d_, m_ in product(data, model):
    plot(d_, m_, big_group[d_][m_])


