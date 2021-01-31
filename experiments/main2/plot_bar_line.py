import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams.update({"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]})
sns.set_theme(style="darkgrid")
for d in ['sat', 'u2', 'u4', 'google', 'bats']:
    lm = pd.read_csv('./experiments_results/summary/main2.test.prediction.{}.csv'.format(d))
    we = pd.read_csv('./experiments_results/summary/statistics.test.prediction.{}.csv'.format(d))
    pmi = pd.read_csv('./experiments_results/summary/statistics.test.prediction.pmi.1.{}.csv'.format(d))
    full = pd.concat([lm, we, pmi])
    full['model'] = ['RoBERTa'] * len(lm) + ['FastText'] * len(we) + ['PMI'] * len(we)
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

    if d in ['bats', 'google']:
        plt.xticks(rotation=90)
    else:
        plt.xticks(rotation=60)
    ax = sns.barplot(x='prefix', y='accuracy', hue='model', data=f, order=order, hue_order=['PMI', 'FastText', 'RoBERTa'])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels)
    ax.set_xlabel(None)
    ax.set_ylabel('Accuracy', fontsize=12)
    if d == 'bats':
        ax.tick_params(labelsize=10)
    else:
        ax.tick_params(labelsize=12)
    fig = ax.get_figure()
    plt.tight_layout()
    fig.savefig('./experiments_results/summary/main2_figure/bar.{}.png'.format(d))
    plt.close()

    if d in ['bats', 'google']:
        plt.xticks(rotation=90)
    else:
        plt.xticks(rotation=60)
    if order:
        f['order'] = f['prefix'].apply(lambda x: order.index(x))
    else:
        f['order'] = f['prefix']
    f = f.sort_values(by='order')
    ax = sns.lineplot(x='prefix', y='accuracy', hue='model', data=f, sort=False, hue_order=['PMI', 'FastText', 'RoBERTa'],
                      style="model", markers=True, dashes=[(2, 2), (2, 2), (2, 2)])
    ax.legend(handles=handles, labels=labels)
    ax.set_xlabel(None)
    ax.set_ylabel('Accuracy', fontsize=12)
    if d == 'bats':
        ax.tick_params(labelsize=10)
    else:
        ax.tick_params(labelsize=12)
    fig = ax.get_figure()
    plt.tight_layout()
    fig.savefig('./experiments_results/summary/main2_figure/line.{}.png'.format(d))
    plt.close()
