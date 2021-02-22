import seaborn as sns
import os
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams.update({"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]})
sns.set_theme(style="darkgrid")
model_order = ['PMI', 'FastText', 'BERT', 'GPT2', 'RoBERTa']
export_prefix = 'main2.ppl_pmi'
os.makedirs('./experiments_results/summary/figure', exist_ok=True)
for d in ['sat', 'u2', 'u4', 'google', 'bats']:
    bert = pd.read_csv('./experiments_results/summary/{}.test.prediction.{}.bert-large-cased.csv'.format(export_prefix, d))
    roberta = pd.read_csv('./experiments_results/summary/{}.test.prediction.{}.roberta-large.csv'.format(export_prefix, d))
    gpt = pd.read_csv('./experiments_results/summary/{}.test.prediction.{}.gpt2-xl.csv'.format(export_prefix, d))
    we = pd.read_csv('./experiments_results/summary/statistics.test.prediction.{}.csv'.format(d))
    pmi = pd.read_csv('./experiments_results/summary/statistics.test.prediction.pmi.1.{}.csv'.format(d))
    # gpt = pmi
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
    g = full.groupby(['prefix', 'model']).accuracy.mean() * 100
    f = g.to_frame()
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
        fig.savefig('./experiments_results/summary/figure/{}.bar.{}.png'.format(export_prefix, d))
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
        fig.savefig('./experiments_results/summary/figure/{}.line.{}.png'.format(export_prefix, d))
        plt.close()
