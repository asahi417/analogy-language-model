import alm
import os
import json
import argparse
from glob import glob
from tqdm import tqdm
import pandas as pd

export_dir = './experiments/ppl_pmi_negative/results'
templates = [
    ['what-is-to'],  # u2/u4 first, sat second
    ['she-to-as'],  # u2 second
    ['as-what-same']  # u4 second, sat first
]
# u4 ('p_0', 'min'), u2 ('p_4', 'p_0'), sat ('p_2', 'min')
aggregation_positive = ['min', 'p_0', 'p_4', 'p_2']
aggregation_negatives = ['max', 'mean', 'min', 'p_0', 'p_1', 'p_2', 'p_3', 'p_4', 'p_5', 'p_6', 'p_7',
                         'p_8', 'p_9', 'p_10', 'p_11']
ppl_pmi_aggregation = ['max', 'mean', 'min', 'p_0', 'p_1']
lambdas = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
alphas = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
permutation_negative_weight = [0, 0.2, 0.4, 0.6, 0.8, 1.0]


def get_options():
    parser = argparse.ArgumentParser(description='command line tool to test finetuned NER model',)
    parser.add_argument('-e', '--experiment', default=3, type=int)

    return parser.parse_args()


def main(path_to_data):
    # get accuracy
    for template in templates:
        scorer = alm.RelationScorer(model='roberta-large', max_length=32)
        scorer.analogy_test(
            skip_duplication_check=True,
            scoring_method='ppl_pmi',
            path_to_data=path_to_data,
            template_types=template,
            aggregation_positive=aggregation_positive,
            permutation_negative=True,
            aggregation_negative=aggregation_negatives,
            ppl_pmi_lambda=lambdas,
            ppl_pmi_alpha=alphas,
            ppl_pmi_aggregation=ppl_pmi_aggregation,
            no_inference=True,
            export_dir=export_dir,
            permutation_negative_weight=permutation_negative_weight
        )


if __name__ == '__main__':
    opt = get_options()
    if opt.experiment == 0:
        main(path_to_data='./data/sat_package_v3.jsonl')
    if opt.experiment == 1:
        main(path_to_data='./data/u2_raw.jsonl')
    if opt.experiment == 2:
        main(path_to_data='./data/u4_raw.jsonl')

    # export as a csv
    index = ['model', 'path_to_data', 'scoring_method', 'template_types', 'aggregation_positive',
             'aggregation_negative', 'ppl_pmi_lambda', 'ppl_pmi_alpha', 'ppl_pmi_aggregation',
             'permutation_negative_weight']
    df = pd.DataFrame(index=index + ['accuracy'])

    chunk_n = 150000
    pointer = 0
    chunk_pointer = 0
    for i in tqdm(glob('./{}/outputs/*'.format(export_dir))):
        pointer += 1

        with open(os.path.join(i, 'config.json'), 'r') as f:
            config = json.load(f)

        with open(os.path.join(i, 'accuracy.json'), 'r') as f:
            accuracy = json.load(f)
        if chunk_n == pointer:
            pointer = 0
            df = df.T.sort_values(by=index, ignore_index=True)
            df.to_csv('{}/summary.{}.csv'.format(export_dir, chunk_pointer))

            chunk_pointer += 1
            df = pd.DataFrame(index=index + ['accuracy'])

        df[len(df.T)] = [','.join(config[i]) if type(config[i]) is list else config[i] for i in index] + \
                        [round(accuracy['accuracy'] * 100, 2)]
    if pointer != 0:
        df = df.T.sort_values(by=index, ignore_index=True)
        df.to_csv('{}/summary.{}.csv'.format(export_dir, chunk_pointer))

