import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

export_dir = './experiments/pmi_grid/results'
df_all = pd.read_csv('{}/summary.csv'.format(export_dir), index_col=0)
df_all['cond'] = df_all['pmi_aggregation'] + df_all['aggregation_positive']
export_dir = '{}/figure'.format(export_dir)
if not os.path.exists(export_dir):
    os.makedirs(export_dir, exist_ok=True)


def main(path_to_data):

    data_name = os.path.basename(path_to_data).split('.')[0]
    df = df_all[df_all['path_to_data'] == path_to_data]

    # Line plot with 95% interval
    fig = plt.figure()
    fig.clear()
    sns.set_theme(style="darkgrid")
    sns_plot = sns.lineplot(x="pmi_lambda", y="accuracy", data=df)
    sns_plot.set_xlabel("Lambda", fontsize=15)
    sns_plot.set_ylabel("Accuracy", fontsize=15)
    sns_plot.tick_params(labelsize=10)
    fig = sns_plot.get_figure()
    plt.tight_layout()
    fig.savefig('{}/plot.mean.{}.pmi_lambda.png'.format(export_dir, data_name))
    fig.clear()

    # Line plot with individual result
    fig = plt.figure()
    fig.clear()
    sns_plot = sns.lineplot(x="pmi_lambda", y="accuracy", data=df, hue='cond', legend=None)
    sns_plot.set_xlabel("Lambda", fontsize=15)
    sns_plot.set_ylabel("Accuracy", fontsize=15)
    sns_plot.tick_params(labelsize=10)
    fig = sns_plot.get_figure()
    plt.tight_layout()
    fig.savefig('{}/plot.all.{}.pmi_lambda.png'.format(export_dir, data_name))
    fig.clear()


if __name__ == '__main__':
    main(path_to_data='./data/sat_package_v3.jsonl')
    main(path_to_data='./data/u2.jsonl')
    main(path_to_data='./data/u4.jsonl')
