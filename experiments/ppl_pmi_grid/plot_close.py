import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

export_dir = './experiments/ppl_pmi_grid/results'
df_main = pd.read_csv('{}/summary.close.csv'.format(export_dir), index_col=0)
export_dir = '{}/figure'.format(export_dir)
if not os.path.exists(export_dir):
    os.makedirs(export_dir, exist_ok=True)


def main(path_to_data, ppl_pmi_aggregation, aggregation_positive):
    data_name = os.path.basename(path_to_data).split('.')[0]

    df = df_main[df_main['path_to_data'] == path_to_data]
    df = df[df['ppl_pmi_aggregation'] == ppl_pmi_aggregation][df['aggregation_positive'] == aggregation_positive]

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
    fig.savefig('{}/plot.heatmap.{}.{}.{}.close.png'.format(
        export_dir, data_name, aggregation_positive, ppl_pmi_aggregation))


if __name__ == '__main__':
    main(path_to_data='./data/sat_package_v3.jsonl', ppl_pmi_aggregation='p_1', aggregation_positive='p_2')
    main(path_to_data='./data/u2.jsonl', ppl_pmi_aggregation='p_0', aggregation_positive='p_0')
    main(path_to_data='./data/u4.jsonl', ppl_pmi_aggregation='max', aggregation_positive='p_0')
