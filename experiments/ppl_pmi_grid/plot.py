import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy

export_dir_root = './experiments/ppl_pmi_grid/results'
export_dir = '{}/figure'.format(export_dir_root)
if not os.path.exists(export_dir):
    os.makedirs(export_dir, exist_ok=True)

target_list = {
    'ppl_pmi_lambda': ['Lambda', 1],
    'ppl_pmi_alpha': ['Alpha', 0],
    'permutation_negative_weight': ['Beta', 0]
}  # only upto three

condition = ['aggregation_positive', 'aggregation_negative', 'ppl_pmi_aggregation', 'template_types']


def main(path_to_data):
    data_name = os.path.basename(path_to_data).split('.')[0]
    df_main = pd.read_csv('{}/summary.{}.csv'.format(export_dir_root, data_name), index_col=0)

    print(df_main.columns)
    # df_main['parameters'] = sum([df_main[k] for k in target_list.keys()])

    df_main = df_main[df_main['path_to_data'] == path_to_data]
    sns.set_theme(style="darkgrid")

    for k, (label, default) in target_list.items():
        df = deepcopy(df_main)
        for k_, (_, default_) in target_list.items():
            if k_ != k:
                df = df[df[k_] == default_]

        # Line plot with 95% interval
        fig = plt.figure()
        fig.clear()
        sns_plot = sns.lineplot(x=k, y="accuracy", data=df)
        sns_plot.set_xlabel(label, fontsize=15)
        sns_plot.set_ylabel("Accuracy", fontsize=15)
        sns_plot.tick_params(labelsize=10)
        fig = sns_plot.get_figure()
        plt.tight_layout()
        fig.savefig('{}/plot.mean.{}.{}.png'.format(export_dir, data_name, k))
        fig.clear()

        df = deepcopy(df_main)
        df = df.sort_values(by=['accuracy'], ascending=False)
        df = df[df[k] == target_list[k][1]]
        df_best = df.head(1)
        # use the best setting
        for c in condition:
            df = df[df[c] == df_best[c].values[0]]

        fig = plt.figure()
        fig.clear()
        i, c = [k_ for k_ in target_list.keys() if k != k_]
        result = df.pivot(index=i, columns=c, values='accuracy')
        # print(result.columns.name, result.index.name)
        # input()
        sns_plot = sns.heatmap(result, annot=True, fmt="g", cmap='viridis', cbar=False)
        sns_plot.set_xlabel(target_list[c][0], fontsize=15)
        sns_plot.set_ylabel(target_list[i][0], fontsize=15)
        sns_plot.tick_params(labelsize=10)
        fig = sns_plot.get_figure()
        plt.grid()
        plt.tight_layout()
        fig.savefig('{}/plot.heatmap.{}.{}={}.png'.format(export_dir, data_name, k, target_list[k][1]))


if __name__ == '__main__':
    main(path_to_data='./data/sat_package_v3.jsonl')
    main(path_to_data='./data/u2_raw.jsonl')
    main(path_to_data='./data/u4_raw.jsonl')
