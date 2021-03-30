import seaborn as sns
import matplotlib.pyplot as plt
import alm
import json
import os
plt.rcParams.update({"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]})
os.makedirs('./experiments_results/summary/figure', exist_ok=True)
sns.set_theme(style="darkgrid")
SKIP_BOX_PLOT = False


def plot_box(df_, s, d):
    fig = plt.figure()
    fig.clear()
    if s == 'negative_permutation_aggregation':
        ax = sns.boxplot(x=s, y='accuracy', data=df_, hue='model', hue_order=model,
                         order=[r'val$_{}{}{}$'.format('{', n, '}') for n in range(1, 13)])
                         # order=[r'val$_{}{}{}$'.format('{', n, '}') for n in range(1, 13)] + ['max', 'mean', 'min'])
        plt.xticks(rotation=30)
    elif s == 'positive_permutation_aggregation':
        ax = sns.boxplot(x=s, y='accuracy', data=df_, hue='model', hue_order=model,
                         # order=[r'val$_{}{}{}$'.format('{', n, '}') for n in range(1, 9)] + ['max', 'mean', 'min'])
                         order=[r'val$_{}{}{}$'.format('{', n, '}') for n in range(1, 9)])
    elif s == 'ppl_pmi_aggregation':
        ax = sns.boxplot(x=s, y='accuracy', data=df_, hue='model', hue_order=model,
                         order=[r'val$_{}{}{}$'.format('{', n, '}') for n in range(1, 3)])
                         # order=[r'val$_{}{}{}$'.format('{', n, '}') for n in range(1, 3)] + ['max', 'mean', 'min'])
    else:
        ax = sns.boxplot(x=s, y='accuracy', data=df_, hue='model')

    plt.setp(ax.get_legend().get_texts(), fontsize='15')

    # ax.set_xlabel(n, fontsize=15)
    ax.set_xlabel(None)
    ax.set_ylabel('Relative Improvement', fontsize=15)
    ax.tick_params(labelsize=15)
    fig = ax.get_figure()
    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    labels = [i.replace('bert-large-cased', 'BERT').replace('roberta-large', 'RoBERTa').replace('gpt2-xl', 'GPT2') for i
              in labels]
    ax.legend(handles=handles, labels=labels, loc='upper right')
    fig.savefig('./experiments_results/summary/figure/{}.box.{}.{}.png'.format(export_prefix, d, s))
    plt.close()


if not SKIP_BOX_PLOT:
    df = alm.get_report(export_prefix='experiment.ppl_variants')
    df = df[df.scoring_method == 'ppl_based_pmi']
    df['accuracy'] = df['accuracy'].round(3) * 100
    df['ppl_based_pmi_aggregation'] = df['ppl_based_pmi_aggregation'].apply(lambda x: r'val$_{0}{1}{2}$'.format(
        '{', (int(x.replace('index_', '')) + 1), '}') if 'index' in x else x)
    df['negative_permutation_aggregation'] = df['negative_permutation_aggregation'].apply(lambda x: r'val$_{0}{1}{2}$'.format(
        '{', (int(x.replace('index_', '')) + 1), '}') if 'index' in x else x)
    df['positive_permutation_aggregation'] = df['positive_permutation_aggregation'].apply(lambda x: r'val$_{0}{1}{2}$'.format(
        '{', (int(x.replace('index_', '')) + 1), '}') if 'index' in x else x)
    data = ['sat', 'u2', 'u4', 'google', 'bats']
    model = ['bert-large-cased', 'gpt2-xl', 'roberta-large']
    df = df.sort_values(by=['model'])
    g = df.groupby(['data'])
    mean_acc = json.loads(g.accuracy.mean().to_json())
    df['accuracy_mean'] = df['data'].apply(lambda x: mean_acc[x])
    df['accuracy'] = df['accuracy'] - df['accuracy_mean']


    for s_tmp in ['positive_permutation_aggregation', 'negative_permutation_aggregation', 'ppl_pmi_aggregation']:
        plot_box(df, s_tmp, 'all')
        for data_ in data:
            df_tmp = df[df.data == data_]
            plot_box(df_tmp, s_tmp, data_)

