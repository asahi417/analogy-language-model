import seaborn as sns
import matplotlib.pyplot as plt
import alm
import os
from itertools import product
plt.rcParams.update({"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]})

os.makedirs('./experiments_results/summary/main2_figure', exist_ok=True)
export_prefix = 'main2'
df = alm.get_report(export_prefix=export_prefix)
df['accuracy'] = df['accuracy'].round(3) * 100
data = ['sat', 'u2', 'u4', 'google', 'bats']
model = ['roberta-large', 'gpt2-xl']
big_group = df.groupby(['data', 'model', 'ppl_pmi_alpha', 'negative_permutation_weight']).accuracy.max()

sns.set_theme(style="darkgrid")
for d, m in product(data, model):
    fig = plt.figure()
    fig.clear()
    accuracy = big_group[d][m]
    accuracy = accuracy - accuracy[0.0][0.0]
    accuracy = accuracy.to_frame()
    accuracy.reset_index(inplace=True)
    accuracy = accuracy.pivot(index='ppl_pmi_alpha', columns='negative_permutation_weight', values='accuracy')
    accuracy = accuracy.reindex([0.4, 0.2, 0.0, -0.2, -0.4])
    sns_plot = sns.heatmap(accuracy, annot=True, fmt="g", cbar=False, annot_kws={"fontsize": 15})
    sns_plot.set_xlabel(r'$\beta$', fontsize=15)
    sns_plot.set_ylabel(r'$\alpha$', fontsize=15)
    sns_plot.tick_params(labelsize=12)
    fig = sns_plot.get_figure()
    plt.tight_layout()
    fig.savefig('./experiments_results/summary/main2_figure/heatmap.alpha_beta.{}.{}.png'.format(d, m))
    plt.close()

