import alm
import os
import json
from glob import glob
import pandas as pd

export_dir = './experiments/pmi_grid/results'


def main(path_to_data, template):
    pmi_aggregations = ['max', 'mean', 'min', 'p_0', 'p_1', 'p_2', 'p_3', 'p_4', 'p_5', 'p_6', 'p_7', 'p_8',
                        'p_9', 'p_10', 'p_11']
    aggregation_positives = ['max', 'mean', 'min', 'p_0', 'p_1', 'p_2', 'p_3', 'p_4', 'p_5', 'p_6', 'p_7']
    # get accuracy
    scorer = alm.RelationScorer(model='roberta-large', max_length=32)
    for i in [-2.0, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2.0]:
        for pmi_aggregation in pmi_aggregations:
            list(map(lambda x: scorer.analogy_test(
                scoring_method='pmi',
                pmi_aggregation=pmi_aggregation,
                path_to_data=path_to_data,
                template_types=[template],
                aggregation_positive=x,
                pmi_lambda=i,
                no_inference=True,
                export_dir=export_dir), aggregation_positives))

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


if __name__ == '__main__':
    main(path_to_data='./data/sat_package_v3.jsonl', template='rel-same')
    main(path_to_data='./data/u2.jsonl', template='rel-same')
    main(path_to_data='./data/u4.jsonl', template='what-is-to')
