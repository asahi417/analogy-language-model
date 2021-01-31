import seaborn as sns
import matplotlib.pyplot as plt
import alm
import os
plt.rcParams.update({"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]})
os.makedirs('./experiments_results/summary/main2_figure', exist_ok=True)
export_prefix = 'main2'
df = alm.get_report(export_prefix=export_prefix)
df['accuracy'] = df['accuracy'].round(3) * 100
df['ppl_pmi_aggregation'] = df['ppl_pmi_aggregation'].apply(lambda x: r'val$_{0}{1}{2}$'.format(
    '{', (int(x.replace('index_', '')) + 1), '}') if 'index' in x else x)
df['negative_permutation_aggregation'] = df['negative_permutation_aggregation'].apply(lambda x: r'val$_{0}{1}{2}$'.format(
    '{', (int(x.replace('index_', '')) + 1), '}') if 'index' in x else x)
df['positive_permutation_aggregation'] = df['positive_permutation_aggregation'].apply(lambda x: r'val$_{0}{1}{2}$'.format(
    '{', (int(x.replace('index_', '')) + 1), '}') if 'index' in x else x)
data = ['sat', 'u2', 'u4', 'google', 'bats']
model = ['roberta-large', 'gpt2-xl']
df = df[df.model != 'bert-large-cased']

sns.set_theme(style="darkgrid")
for d in data:
    for s, n in zip(
            ['ppl_pmi_alpha', 'negative_permutation_weight', 'positive_permutation_aggregation',
             'negative_permutation_aggregation', 'ppl_pmi_aggregation'],
            [r'$\alpha$', r'$\beta$', 'Positive permutation', 'Negative permutation', 'PMI permutation']):
        fig = plt.figure()
        fig.clear()
        if s == 'negative_permutation_aggregation':
            ax = sns.boxplot(x=s, y='accuracy', data=df, hue='model',
                             order=[r'val$_{}{}{}$'.format('{', n, '}') for n in range(1, 13)] + ['max', 'mean', 'min'])
        elif s == 'positive_permutation_aggregation':
            ax = sns.boxplot(x=s, y='accuracy', data=df, hue='model',
                             order=[r'val$_{}{}{}$'.format('{', n, '}') for n in range(1, 9)] + ['max', 'mean', 'min'])
        elif s == 'ppl_pmi_aggregation':
            ax = sns.boxplot(x=s, y='accuracy', data=df, hue='model',
                             order=[r'val$_{}{}{}$'.format('{', n, '}') for n in range(1, 3)] + ['max', 'mean', 'min'])
        else:
            ax = sns.boxplot(x=s, y='accuracy', data=df, hue='model')
        handles, labels = ax.get_legend_handles_labels()
        labels = [i.replace('roberta-large', 'RoBERTa').replace('gpt2-xl', 'GPT2') for i in labels]
        ax.legend(handles=handles, labels=labels)
        ax.set_xlabel(n, fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.tick_params(labelsize=10)
        fig = ax.get_figure()
        plt.tight_layout()
        fig.savefig('./experiments_results/summary/main2_figure/box.{}.{}.png'.format(d, s))
        plt.close()
