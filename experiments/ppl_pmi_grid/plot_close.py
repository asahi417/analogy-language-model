import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main(export_dir):
    df = pd.read_csv('{}/summary.csv'.format(export_dir), index_col=0)
    sns.set_theme(style="darkgrid")
    df['aggregation_positive'] = ['P'+i.replace('p_', '') if 'p_' in i else i
                                  for i in df['aggregation_positive'].values.tolist()]

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


if __name__ == '__main__':
    main(export_dir='./experiments/ppl_pmi_grid_close/results')
    main(export_dir='./experiments/ppl_pmi_grid_close/results_u2')
    main(export_dir='./experiments/ppl_pmi_grid_close/results_u4')
