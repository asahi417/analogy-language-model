import seaborn as sns
import matplotlib.pyplot as plt
import alm
import os
from itertools import product

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
## for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})
os.makedirs('./experiments_results/summary/main2_figure', exist_ok=True)
export_prefix = 'main2'
df = alm.get_report(export_prefix=export_prefix)
df['accuracy'] = df['accuracy'].round(4) * 100
data = ['sat', 'u2', 'u4', 'google', 'bats']
model = ['roberta-large', 'gpt2-xl', 'bert-large-cased']
big_group = df.groupby(['data', 'model', 'ppl_pmi_alpha', 'negative_permutation_weight']).accuracy.max()
big_group_perm = df.groupby(['data', 'model', 'positive_permutation_aggregation', 'negative_permutation_aggregation']).accuracy.max()

sns.set_theme(style="darkgrid")
for d, m in product(data, model):
    fig = plt.figure()
    fig.clear()
    # accuracy = group.accuracy.max()
    accuracy = big_group[d][m]
    # print('size per category', group.size())
    accuracy = accuracy - accuracy[0.0][0.0]
    # print('accuracy per category', accuracy)
    accuracy = accuracy.to_frame()
    accuracy.reset_index(inplace=True)
    accuracy = accuracy.pivot(index='ppl_pmi_alpha', columns='negative_permutation_weight', values='accuracy')
    sns_plot = sns.heatmap(accuracy, annot=True, fmt="g", cmap='viridis', cbar=False)

    # sns_plot.set_xlabel('e', fontsize=15)
    sns_plot.set_xlabel(r'$\eta$', fontsize=15)

    # sns_plot.set_ylabel('a', fontsize=15)
    sns_plot.set_ylabel(r'$\alpha$', fontsize=15)
    sns_plot.tick_params(labelsize=10)
    fig = sns_plot.get_figure()
    plt.tight_layout()
    fig.savefig('./experiments_results/summary/main2_figure/heatmap.alpha_eta.{}.{}.png'.format(d, m))


    fig = plt.figure()
    fig.clear()
    # accuracy = group.accuracy.max()
    accuracy = big_group_perm[d][m]
    # print('size per category', group.size())
    accuracy = accuracy - accuracy['index_0']['index_0']
    # print('accuracy per category', accuracy)
    accuracy = accuracy.to_frame()
    accuracy.reset_index(inplace=True)
    accuracy = accuracy.pivot(index='positive_permutation_aggregation', columns='negative_permutation_aggregation', values='accuracy')
    sns_plot = sns.heatmap(accuracy, annot=True, fmt="g", cmap='viridis', cbar=False)

    # sns_plot.set_xlabel('e', fontsize=15)
    sns_plot.set_xlabel(r'$\eta$', fontsize=15)

    # sns_plot.set_ylabel('a', fontsize=15)
    sns_plot.set_ylabel(r'$\alpha$', fontsize=15)
    sns_plot.tick_params(labelsize=10)
    fig = sns_plot.get_figure()
    plt.tight_layout()
    fig.savefig('./experiments_results/summary/main2_figure/heatmap.permutations.{}.{}.png'.format(d, m))


