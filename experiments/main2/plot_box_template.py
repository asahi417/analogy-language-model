import seaborn as sns
import matplotlib.pyplot as plt
import alm
import json
import os
plt.rcParams.update({"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]})
os.makedirs('./experiments_results/summary/main2_figure', exist_ok=True)
export_prefix = 'main2'
df = alm.get_report(export_prefix=export_prefix)
df['accuracy'] = df['accuracy'].round(3) * 100

# TEMPLATES = {
#     'is-to-what': r"{\it to what}",
#     'is-to-as': r"{\it to as}",
#     'rel-same': r'{\it rel same}',
#     'what-is-to': r'{\it what to}',
#     'she-to-as': r'{\it she as}',
#     'as-what-same': r'{\it as what}'
# }
TEMPLATES = {
    'is-to-as': "to-as",
    'is-to-what': "to-what",
    'rel-same': 'rel-same',
    'what-is-to': 'what-to',
    'she-to-as': 'she-as',
    'as-what-same': 'as-what'
}

df['template_type'] = df['template_type'].apply(lambda x: TEMPLATES[x])
data = ['sat', 'u2', 'u4', 'google', 'bats']
model = ['bert-large-cased', 'gpt2-xl', 'roberta-large']
df = df.sort_values(by=['model'])
g = df.groupby(['data'])
mean_acc = json.loads(g.accuracy.mean().to_json())
df['accuracy_mean'] = df['data'].apply(lambda x: mean_acc[x])
df['accuracy'] = df['accuracy'] - df['accuracy_mean']

sns.set_theme(style="darkgrid")


def plot(df_, d):
    fig = plt.figure()
    fig.clear()
    plt.xticks(rotation=30)
    ax = sns.boxplot(x='template_type', y='accuracy', data=df_, hue='model', hue_order=model, order=list(TEMPLATES.values()))
    plt.setp(ax.get_legend().get_texts(), fontsize='15')
    ax.set_xlabel(None)
    ax.set_ylabel('Relative Improvement', fontsize=15)
    ax.tick_params(labelsize=15)
    fig = ax.get_figure()
    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    labels = [
        i.replace('bert-large-cased', 'BERT').replace('roberta-large', 'RoBERTa').replace('gpt2-xl', 'GPT2') for i
        in labels]
    ax.legend(handles=handles, labels=labels, loc='upper right')
    fig.savefig('./experiments_results/summary/main2_figure/box.{}.template_type.png'.format(d))
    plt.close()


plot(df, 'all')
for data_ in data:
    df_tmp = df[df.data == data_]
    plot(df_tmp, data_)
