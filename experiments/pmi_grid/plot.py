import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


export_dir = './experiments/pmi_grid/results'
df = pd.read_csv('{}/summary.csv'.format(export_dir), index_col=0)
df['cond'] = df['pmi_aggregation'] + df['aggregation_positive']

# Line plot with 95% interval
sns.set_theme(style="darkgrid")
sns_plot = sns.lineplot(x="pmi_lambda", y="accuracy", data=df)
sns_plot.set_xlabel("Lambda", fontsize=15)
sns_plot.set_ylabel("Accuracy", fontsize=15)
sns_plot.tick_params(labelsize=10)
fig = sns_plot.get_figure()
plt.tight_layout()
fig.savefig('{}/plot.mean.png'.format(export_dir))

# # Line plot with individual result
sns_plot = sns.lineplot(x="pmi_lambda", y="accuracy", data=df, hue='cond', legend=None)
sns_plot.set_xlabel("Lambda", fontsize=15)
sns_plot.set_ylabel("Accuracy", fontsize=15)
sns_plot.tick_params(labelsize=10)
fig = sns_plot.get_figure()
plt.tight_layout()
fig.savefig('{}/plot.png'.format(export_dir))

