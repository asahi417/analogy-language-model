import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

export_dir_root = './experiments/ppl_pmi_grid/results'

df_main = pd.read_csv('{}/summary.csv'.format(export_dir_root), index_col=0)


def main(path_to_data, ppl_pmi_aggregation=None, aggregation_positive=None):
    data_name = os.path.basename(path_to_data).split('.')[0]
    export_dir = '{}/figure'.format(export_dir_root)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=True)

    df = df_main[df_main['path_to_data'] == path_to_data]
    df['aggregations'] = df['aggregation_positive'] + df['ppl_pmi_aggregation']
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
        plt.tight_layout()
        fig.savefig('{}/plot.mean.{}.{}.png'.format(export_dir, data_name, i))
        fig.clear()

        # Line plot with individual result
        fig = plt.figure()
        fig.clear()
        sns_plot = sns.lineplot(x=i, y="accuracy", data=tmp_df, hue='aggregations')
        sns_plot.set_xlabel(n, fontsize=15)
        sns_plot.set_ylabel("Accuracy", fontsize=15)
        sns_plot.tick_params(labelsize=10)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        fig = sns_plot.get_figure()
        plt.tight_layout()
        fig.savefig('{}/plot.{}.png'.format(export_dir, i))
        fig.clear()

    for i in list(set(list(df['aggregation_positive'].values))):
        for _i in list(set(list(df['ppl_pmi_aggregation'].values))):
            if ppl_pmi_aggregation is not None and _i != ppl_pmi_aggregation:
                continue
            if aggregation_positive is not None and i != aggregation_positive:
                continue

            fig = plt.figure()
            fig.clear()
            tmp = df[df['aggregation_positive'] == i][df['ppl_pmi_aggregation'] == _i]
            result = tmp.pivot(index='ppl_pmi_lambda', columns='ppl_pmi_alpha', values='accuracy')
            sns_plot = sns.heatmap(result, annot=True, fmt="g", cmap='viridis', cbar=False)
            sns_plot.set_xlabel("Alpha", fontsize=15)
            sns_plot.set_ylabel("Lambda", fontsize=15)
            sns_plot.tick_params(labelsize=10)
            fig = sns_plot.get_figure()
            plt.tight_layout()
            fig.savefig('{}/plot.heatmap.{}.{}.png'.format(export_dir, i, _i))


if __name__ == '__main__':
    main(path_to_data='./data/sat_package_v3.jsonl', ppl_pmi_aggregation='p_1', aggregation_positive='p_2')
    main(path_to_data='./data/u2.jsonl', ppl_pmi_aggregation='p_0', aggregation_positive='p_0')
    main(path_to_data='./data/u4.jsonl', ppl_pmi_aggregation='max', aggregation_positive='p_0')
