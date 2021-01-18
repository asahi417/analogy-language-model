import alm
import os
import json
from glob import glob
import pandas as pd

export_dir = './experiments/pmi_grid/results'
templates = [
    ['what-is-to'],  # u2/u4 first, sat second
    ['rel-same'],  # u2 second
    ['as-what-same']  # u4 second, sat first
]
pmi_aggregations = ['max', 'mean', 'min', 'p_0', 'p_1', 'p_2', 'p_3', 'p_4', 'p_5', 'p_6', 'p_7', 'p_8',
                        'p_9', 'p_10', 'p_11']
aggregation_positives = ['max', 'mean', 'min', 'p_0', 'p_1', 'p_2', 'p_3', 'p_4', 'p_5', 'p_6', 'p_7']
pmi_lambdas = [-2.0, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2.0]


def main(path_to_data):
    # get accuracy
    scorer = alm.RelationScorer(model='roberta-large', max_length=32)
    for pmi_lambda in pmi_lambdas:
        for template in templates:
            list(map(lambda x: scorer.analogy_test(
                scoring_method='pmi',
                pmi_aggregation=x,
                path_to_data=path_to_data,
                template_types=template,
                aggregation_positive=aggregation_positives,
                pmi_lambda=pmi_lambda,
                no_inference=True,
                export_dir=export_dir), pmi_aggregations))


if __name__ == '__main__':
    main(path_to_data='./data/sat_package_v3.jsonl')
    main(path_to_data='./data/u4_raw.jsonl')
    main(path_to_data='./data/u2_raw.jsonl')

    # export as a csv
    index = ['model', 'max_length', 'path_to_data', 'scoring_method', 'template_types', 'aggregation_positive',
             'pmi_aggregation', 'pmi_lambda']
    df = pd.DataFrame(index=index + ['accuracy'])

    for i in glob('{}/outputs/*'.format(export_dir)):
        with open(os.path.join(i, 'accuracy.json'), 'r') as f:
            accuracy = json.load(f)
        with open(os.path.join(i, 'config.json'), 'r') as f:
            config = json.load(f)
        df[len(df.T)] = [','.join(config[i]) if type(config[i]) is list else config[i] for i in index] + \
                        [round(accuracy['accuracy'] * 100, 2)]

    df = df.T
    df = df.sort_values(by=index, ignore_index=True)
    df.to_csv('{}/summary.csv'.format(export_dir))

