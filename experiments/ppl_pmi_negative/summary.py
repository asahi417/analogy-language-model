import alm
import os
import json
from glob import glob
import pandas as pd

export_dir = './experiments/ppl_pmi_negative/results'
aggregation_negatives = ['max', 'mean', 'min', 'p_0', 'p_1', 'p_2', 'p_3', 'p_4', 'p_5', 'p_6', 'p_7',
                         'p_8', 'p_9', 'p_10', 'p_11']
aggregation_ppl_pmi = ['max', 'mean', 'min', 'p_0', 'p_1']
lambdas = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
alphas = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
permutation_negative_weight = [0, 0.2, 0.4, 0.6, 0.8, 1.0]


def main(path_to_data, template, aggregation_positive):
    # get accuracy
    scorer = alm.RelationScorer(model='roberta-large', max_length=32)
    scorer.analogy_test(
        scoring_method='ppl_pmi',
        path_to_data=path_to_data,
        template_types=[template],
        aggregation_positive=aggregation_positive,
        permutation_negative=True,
        aggregation_negative=aggregation_negatives,
        ppl_pmi_lambda=lambdas,
        ppl_pmi_alpha=alphas,
        ppl_pmi_aggregation=aggregation_ppl_pmi,
        no_inference=True,
        export_dir=export_dir,
        permutation_negative_weight=permutation_negative_weight
    )


if __name__ == '__main__':
    ind = int(os.getenv('EXPERIMENT', '0'))
    if ind == 0:
        main(path_to_data='./data/sat_package_v3.jsonl', template='as-what-same', aggregation_positive='p_2')
    if ind == 1:
        main(path_to_data='./data/u2.jsonl', template='she-to-as', aggregation_positive='min')
    if ind == 2:
        main(path_to_data='./data/u4.jsonl', template='what-is-to', aggregation_positive='p_0')

    # export as a csv
    index = ['model', 'path_to_data', 'scoring_method', 'template_types', 'aggregation_positive',
             'aggregation_negative', 'ppl_pmi_lambda', 'ppl_pmi_alpha', 'ppl_pmi_aggregation']
    df = pd.DataFrame(index=index + ['accuracy'])

    for i in glob('./{}/outputs/*'.format(export_dir)):
        with open(os.path.join(i, 'config.json'), 'r') as f:
            config = json.load(f)
            if config['ppl_pmi_lambda'] not in lambdas or config['ppl_pmi_alpha'] not in alphas:
                print(config['ppl_pmi_lambda'], config['ppl_pmi_alpha'])
                continue

        with open(os.path.join(i, 'accuracy.json'), 'r') as f:
            accuracy = json.load(f)
        df[len(df.T)] = [','.join(config[i]) if type(config[i]) is list else config[i] for i in index] + \
                        [round(accuracy['accuracy'] * 100, 2)]

    df = df.T
    df = df.sort_values(by=index, ignore_index=True)
    df.to_csv('{}/summary.csv'.format(export_dir))
