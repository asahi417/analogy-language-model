import seaborn as sns
import matplotlib.pyplot as plt
import alm
import os
plt.rcParams.update({
                "text.usetex": True,
                "font.family": "sans-serif",
                "font.sans-serif": ["Helvetica"]})
os.makedirs('./experiments_results/summary/main2_figure', exist_ok=True)
export_prefix = 'main2'
df = alm.get_report(export_prefix=export_prefix)
df['accuracy'] = df['accuracy'].round(3) * 100
df['ppl_pmi_aggregation'] = df['ppl_pmi_aggregation'].apply(lambda x: x.replace('index_', 'P'))
df['negative_permutation_aggregation'] = df['negative_permutation_aggregation'].apply(lambda x: x.replace('index_', 'P'))
df['positive_permutation_aggregation'] = df['positive_permutation_aggregation'].apply(lambda x: x.replace('index_', 'P'))
data = ['sat', 'u2', 'u4', 'google', 'bats']
model = ['roberta-large', 'gpt2-xl']
df = df[df.model != 'bert-large-cased']

sns.set_theme(style="darkgrid")
for d in data:
    for s, n in zip(
            ['ppl_pmi_alpha', 'negative_permutation_weight', 'positive_permutation_aggregation',
             'negative_permutation_aggregation', 'ppl_pmi_aggregation'],
            [r'$\alpha$', r'$\beta$', 'positive permutation', 'negative permutation', 'PMI permutation']):

        fig = plt.figure()
        fig.clear()
        ax = sns.boxplot(x=s, y='accuracy', data=df, hue='model')
        handles, labels = ax.get_legend_handles_labels()
        labels = [i.replace('roberta-large', 'RoBERTa').replace('gpt2-xl', 'GPT2') for i in labels]
        ax.legend(handles=handles, labels=labels)
        ax.set_xlabel(n, fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.tick_params(labelsize=10)
        fig = ax.get_figure()
        plt.tight_layout()
        fig.savefig('./experiments_results/summary/main2_figure/box.alpha_beta.{}.{}.png'.format(d, s))
        plt.close()
