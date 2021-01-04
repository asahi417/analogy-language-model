import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

export_dir = './results_ppl_pmi_tuning'
df = pd.read_csv('./{}/summary.csv'.format(export_dir), index_col=0)
df['aggregation_positive'] = ['P'+i.replace('p_', '') if 'p_' in i else i
                                  for i in df['aggregation_positive'].values.tolist()]
sns.set_theme(style="darkgrid")

for i, n in zip(['ppl_pmi_lambda', 'ppl_pmi_alpha'], ['Lambda', 'Alpha']):
    if i == 'ppl_pmi_lambda':
        tmp_df = df[df['ppl_pmi_alpha'] == 1]
    else:
        tmp_df = df[df['ppl_pmi_lambda'] == 1]

    print(tmp_df)

    # Line plot with 95% interval
    fig = plt.figure()
    fig.clear()
    sns_plot = sns.lineplot(x=i, y="accuracy", data=tmp_df)
    sns_plot.set_xlabel(n, fontsize=15)
    sns_plot.set_ylabel("Accuracy", fontsize=15)
    sns_plot.tick_params(labelsize=10)
    fig = sns_plot.get_figure()
    fig.savefig('{}/plot_mean.{}.png'.format(export_dir, i))
    fig.clear()

    # Line plot with individual result
    fig = plt.figure()
    fig.clear()
    sns_plot = sns.lineplot(x=i, y="accuracy", data=tmp_df, hue='aggregation_positive')
    sns_plot.set_xlabel(n, fontsize=15)
    sns_plot.set_ylabel("Accuracy", fontsize=15)
    sns_plot.tick_params(labelsize=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    fig = sns_plot.get_figure()
    fig.savefig('{}/plot.{}.png'.format(export_dir, i))
    fig.clear()

for i in list(set(list(df['aggregation_positive'].values))):
    print(i)

    tmp = df[df['aggregation_positive'] == i]
    result = tmp.pivot(index='ppl_pmi_lambda', columns='ppl_pmi_alpha', values='accuracy')
    sns_plot = sns.heatmap(result, annot=True, fmt="g", cmap='viridis')
    sns_plot.set_xlabel("Lambda", fontsize=15)
    sns_plot.set_ylabel("Alpha", fontsize=15)
    sns_plot.tick_params(labelsize=10)
    fig = sns_plot.get_figure()
    fig.savefig('{}/plot.heatmap.{}.png'.format(export_dir, i))


