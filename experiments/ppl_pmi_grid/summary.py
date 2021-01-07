import alm
import os
import json
from glob import glob
import pandas as pd

aggregation_positives = ['max', 'mean', 'min', 'p_0', 'p_1', 'p_2', 'p_3', 'p_4', 'p_5', 'p_6', 'p_7']
aggregation_ppl_pmi = ['max', 'mean', 'min', 'p_0', 'p_1']
export_dir = './experiments/ppl_pmi_grid/results'
lambdas = [-2.0, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2.0]
alphas = [-2.0, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2.0]


def main(path_to_data, template):
    # get accuracy
    scorer = alm.RelationScorer(model='roberta-large', max_length=32)
    for i in lambdas:
        for _i in alphas:
            for __i in aggregation_ppl_pmi:
                list(map(lambda x: scorer.analogy_test(
                    scoring_method='ppl_pmi',
                    path_to_data=path_to_data,
                    template_types=[template],
                    aggregation_positive=x,
                    ppl_pmi_lambda=i,
                    ppl_pmi_alpha=_i,
                    ppl_pmi_aggregation=__i,
                    no_inference=True,
                    overwrite_output=False,
                    export_dir=export_dir
                    ),  aggregation_positives))


if __name__ == '__main__':
    # main(path_to_data='./data/sat_package_v3.jsonl', template='as-what-same')  # p_2
    # main(path_to_data='./data/u2.jsonl', template='she-to-as')  # min
    # main(path_to_data='./data/u4.jsonl', template='what-is-to')  # p_0

    # export as a csv
    index = ['model', 'path_to_data', 'scoring_method', 'template_types', 'aggregation_positive', 'ppl_pmi_lambda',
             'ppl_pmi_alpha', 'ppl_pmi_aggregation']
    df = pd.DataFrame(index=index + ['accuracy'])

    for i in glob('./{}/outputs/*'.format(export_dir)):
        with open(os.path.join(i, 'config.json'), 'r') as f:
            config = json.load(f)
            if config['ppl_pmi_lambda'] not in lambdas:
                print(config['ppl_pmi_lambda'])
                continue
            if config['ppl_pmi_alpha'] not in alphas:
                print(config['ppl_pmi_alpha'])
                continue

        with open(os.path.join(i, 'accuracy.json'), 'r') as f:
            accuracy = json.load(f)
        df[len(df.T)] = [','.join(config[i]) if type(config[i]) is list else config[i] for i in index] + \
                        [round(accuracy['accuracy'] * 100, 2)]

    df = df.T
    df = df.sort_values(by=index, ignore_index=True)
    df.to_csv('{}/summary.csv'.format(export_dir))
