import seaborn as sns
import matplotlib.pyplot as plt
import alm
import json
import os
import pandas as pd

SKIP_BOX_PLOT = False
SKIP_LINE_PLOT = False

plt.rcParams.update({"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]})
os.makedirs('./experiments_results/summary/figure', exist_ok=True)
sns.set_theme(style="darkgrid")


def plot_box(df_, _m, s, d):
    df_ = df_[df_.scoring_method == _m]
    fig = plt.figure()
    fig.clear()
    if s == 'negative_permutation_aggregation':
        ax = sns.boxplot(x=s, y='accuracy', data=df_, hue='model', hue_order=model,
                         order=[r'val$_{}{}{}$'.format('{', n, '}') for n in range(1, 13)])
        plt.xticks(rotation=30)
    elif s == 'positive_permutation_aggregation':
        ax = sns.boxplot(x=s, y='accuracy', data=df_, hue='model', hue_order=model,
                         order=[r'val$_{}{}{}$'.format('{', n, '}') for n in range(1, 9)])
    elif s == 'ppl_based_pmi_aggregation':
        ax = sns.boxplot(x=s, y='accuracy', data=df_, hue='model', hue_order=model,
                         order=[r'val$_{}{}{}$'.format('{', n, '}') for n in range(1, 3)])
    else:
        ax = sns.boxplot(x=s, y='accuracy', data=df_, hue='model')

    plt.setp(ax.get_legend().get_texts(), fontsize='15')

    ax.set_xlabel(None)
    ax.set_ylabel('Relative Improvement', fontsize=15)
    ax.tick_params(labelsize=15)
    fig = ax.get_figure()
    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    labels = [i.replace('bert-large-cased', 'BERT').replace('roberta-large', 'RoBERTa').replace('gpt2-xl', 'GPT2') for i
              in labels]
    ax.legend(handles=handles, labels=labels, loc='upper right')
    fig.savefig('./experiments_results/summary/figure/box.{}.{}.{}.png'.format(d, _m, s))
    plt.close()


def plot_line():
    we_model = 'fasttext'
    model_order = ['PMI', 'FastText', 'BERT', 'GPT2', 'RoBERTa']

    for d in ['sat', 'u2', 'u4', 'google', 'bats']:
        we = pd.read_csv('./experiments_results/summary/prediction_file/experiment.word_embedding.test.prediction.{}.{}.csv'.format(d, we_model))
        pmi = pd.read_csv('./experiments_results/summary/prediction_file/experiment.pmi.test.prediction.{}.csv'.format(d))
        for m in ['ppl_based_pmi', 'ppl_marginal_bias']:
            bert = pd.read_csv('./experiments_results/summary/prediction_file/experiment.ppl_variants.test.prediction.{}.bert-large-cased.{}.csv'.format(d, m))
            roberta = pd.read_csv('./experiments_results/summary/prediction_file/experiment.ppl_variants.test.prediction.{}.roberta-large.{}.csv'.format(d, m))
            gpt = pd.read_csv('./experiments_results/summary/prediction_file/experiment.ppl_variants.test.prediction.{}.gpt2-xl.{}.csv'.format(d, m))
            full = pd.concat([pmi, we, bert, gpt, roberta])
            full['model'] = ['PMI'] * len(pmi) + ['FastText'] * len(we) + ['BERT'] * len(bert) + ['GPT2'] * len(gpt) + ['RoBERTa'] * len(roberta)
            full['accuracy'] = full['prediction'] == full['answer']
            if d == 'sat':
                full['prefix'] = full['prefix'].apply(lambda x: 'SAT' if 'FROM REAL SAT' in x else 'not SAT')
            elif d == 'bats':
                full['prefix'] = full['prefix'].apply(lambda x: x.split(' [')[-1].split(']')[0].
                                                      replace('_', ' ').replace(' - ', ':').replace(' reg', '').
                                                      replace('V', 'v+').replace(' irreg', ''))
                meta = {
                    'Morphological': [
                        'adj:comparative', 'adj:superlative', 'adj+ly', 'adj+ness', 'verb 3pSg:v+ed', 'verb v+ing:3pSg',
                        'verb v+ing:v+ed', 'verb inf:3pSg', 'verb inf:v+ed', 'verb inf:v+ing', 'verb+able', 'verb+er',
                        'verb+ment', 'verb+tion', 'un+adj', 'noun+less', 'over+adj',
                    ],
                    'Lexical': [
                        'hypernyms:animals', 'hypernyms:misc', 'hyponyms:misc', 'antonyms:binary', 'antonyms:gradable',
                        'meronyms:member', 'meronyms:part', 'meronyms:substance', 'synonyms:exact', 'synonyms:intensity',
                        're+verb'
                    ],
                    'Encyclopedic': [
                        'UK city:county', 'animal:shelter', 'animal:sound', 'animal:young', 'country:capital',
                        'country:language', 'male:female', 'name:nationality', 'name:occupation', 'noun:plural',
                        'things:color',
                    ]
                }
                full['prefix'] = full['prefix'].apply(lambda x: [k for k, v in meta.items() if x in v][0])
            elif d == 'google':
                full['prefix'] = full['prefix'].apply(lambda x: 'Morphological' if 'gram' in x else "Semantic")
            f = (full.groupby(['prefix', 'model']).accuracy.mean() * 100).to_frame()
            f.reset_index(inplace=True)
            order = None

            if d == 'u2':
                order = ['grade{}'.format(i) for i in range(4, 13)]
            elif d == 'u4':
                order = ['high-beginning', 'low-intermediate', 'high-intermediate', 'low-advanced', 'high-advanced']

            if d in ['sat', 'google', 'bats']:
                if d in ['u4', 'u2']:
                    plt.xticks(rotation=15)
                ax = sns.barplot(x='prefix', y='accuracy', hue='model', data=f, order=order,
                                 hue_order=model_order)
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles=handles, labels=labels)
                plt.setp(ax.get_legend().get_texts(), fontsize='15')
                ax.set_xlabel(None)
                ax.set_ylabel('Accuracy', fontsize=15)
                ax.tick_params(labelsize=15)
                fig = ax.get_figure()
                plt.tight_layout()
                fig.savefig('./experiments_results/summary/figure/bar.{}.{}.png'.format(d, m))
                plt.close()

            if d in ['u4', 'u2']:
                if d in ['u4', 'u2']:
                    plt.xticks(rotation=15)
                if order:
                    f['order'] = f['prefix'].apply(lambda x: order.index(x))
                else:
                    f['order'] = f['prefix']
                f = f.sort_values(by=['order', 'model'])
                ax = sns.lineplot(x='prefix', y='accuracy', hue='model', data=f, sort=False,
                                  style="model", markers=True, dashes=[(1, 0), (1, 0), (1, 0), (2, 1), (1, 0)],
                                  hue_order=model_order)
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles=handles, labels=labels)
                plt.setp(ax.get_legend().get_texts(), fontsize='15')
                ax.set_xlabel(None)
                ax.set_ylabel('Accuracy', fontsize=15)
                ax.tick_params(labelsize=15)
                fig = ax.get_figure()
                plt.tight_layout()
                fig.savefig('./experiments_results/summary/figure/line.{}.{}.png'.format(d, m))
                plt.close()


if not SKIP_BOX_PLOT:
    df = alm.get_report(export_prefix='experiment.ppl_variants', test=True)
    df['accuracy'] = df['accuracy'].round(3) * 100
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

    for m in ['ppl_based_pmi', 'ppl_marginal_bias']:
        for s_tmp in ['positive_permutation_aggregation', 'negative_permutation_aggregation']:
            plot_box(df, m, s_tmp, 'all')
            for data_ in data:
                df_tmp = df[df.data == data_]
                plot_box(df_tmp, m, s_tmp, data_)

if not SKIP_LINE_PLOT:
    plot_line()
