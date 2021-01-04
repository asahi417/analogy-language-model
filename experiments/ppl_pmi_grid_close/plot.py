import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

export_dir = './experiments/ppl_pmi_grid_close/results'
df = pd.read_csv('{}/summary.csv'.format(export_dir), index_col=0)
sns.set_theme(style="darkgrid")


fig = plt.figure()
fig.clear()
result = df.pivot(index='ppl_pmi_lambda', columns='ppl_pmi_alpha', values='accuracy')
sns_plot = sns.heatmap(result, annot=True, fmt="g", cmap='viridis', cbar=False)
sns_plot.set_xlabel("Alpha", fontsize=15)
sns_plot.set_ylabel("Lambda", fontsize=15)
sns_plot.tick_params(labelsize=10)
fig = sns_plot.get_figure()
plt.tight_layout()
fig.savefig('{}/plot.heatmap.{}.png'.format(export_dir, df['aggregation_positive'][0]))


