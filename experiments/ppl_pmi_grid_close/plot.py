import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

export_dir = './experiments/ppl_pmi_grid_close/results'
df = pd.read_csv('./{}/summary.csv'.format(export_dir), index_col=0)
df['aggregation_positive'] = ['P'+i.replace('p_', '') if 'p_' in i else i
                                  for i in df['aggregation_positive'].values.tolist()]
sns.set_theme(style="darkgrid")


for i in list(set(list(df['aggregation_positive'].values))):
    fig = plt.figure()
    fig.clear()
    tmp = df[df['aggregation_positive'] == i]
    result = tmp.pivot(index='ppl_pmi_lambda', columns='ppl_pmi_alpha', values='accuracy')
    sns_plot = sns.heatmap(result, annot=True, fmt="g", cmap='viridis', cbar=False)
    sns_plot.set_xlabel("Lambda", fontsize=15)
    sns_plot.set_ylabel("Alpha", fontsize=15)
    sns_plot.tick_params(labelsize=10)
    fig = sns_plot.get_figure()
    fig.savefig('{}/plot.heatmap.{}.png'.format(export_dir, i))


