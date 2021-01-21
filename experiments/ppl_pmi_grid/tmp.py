import alm
import os
import json
import argparse
from glob import glob
import tqdm
from multiprocessing import Pool

import pandas as pd

export_dir = './experiments/ppl_pmi_grid/results'
templates = [
    ['what-is-to'],  # u2/u4 first, sat second
]
# u4 ('p_0', 'min'), u2 ('p_4', 'p_0'), sat ('p_2', 'min')
aggregation_positive = ['p_4']
aggregation_negative = ['min']
ppl_pmi_aggregation = ['mean']
permutation_negative_weight = [0]
index = ['model', 'path_to_data', 'scoring_method', 'template_types', 'aggregation_positive',
         'aggregation_negative', 'ppl_pmi_lambda', 'ppl_pmi_alpha', 'ppl_pmi_aggregation',
         'permutation_negative_weight']
total_files = glob('./{}/outputs/*'.format(export_dir))
pbar = tqdm.tqdm(total=int(len(total_files)/os.cpu_count()))


def get_options():
    parser = argparse.ArgumentParser(description='command line tool to test finetuned NER model',)
    parser.add_argument('-e', '--experiment', default=None, type=int)

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
            aggregation_negative=aggregation_negative,
            ppl_pmi_lambda=0.7,
            ppl_pmi_alpha=0.5,
            ppl_pmi_aggregation=ppl_pmi_aggregation,
            no_inference=True,
            export_dir=export_dir,
            permutation_negative_weight=permutation_negative_weight
        )
        scorer.analogy_test(
            skip_duplication_check=True,
            scoring_method='ppl_pmi',
            path_to_data=path_to_data,
            template_types=template,
            aggregation_positive=aggregation_positive,
            permutation_negative=True,
            aggregation_negative=aggregation_negative,
            ppl_pmi_lambda=0.8,
            ppl_pmi_alpha=0.2,
            ppl_pmi_aggregation=ppl_pmi_aggregation,
            no_inference=True,
            export_dir=export_dir,
            permutation_negative_weight=permutation_negative_weight
        )


def get_result(_file):
    with open(os.path.join(_file, 'config.json'), 'r') as f:
        config = json.load(f)
    with open(os.path.join(_file, 'accuracy.json'), 'r') as f:
        accuracy = json.load(f)
    pbar.update(1)
    return [','.join(config[i]) if type(config[i]) is list else config[i] for i in index] + \
           [round(accuracy['accuracy'] * 100, 2)]


if __name__ == '__main__':
    opt = get_options()
    main(path_to_data='./data/u2_raw.jsonl')
    print('CPU count: {}'.format(os.cpu_count()))
    pool = Pool()  # Create a multiprocessing Pool
    total_files = glob('./{}/outputs/*'.format(export_dir))
    print('total file: {}'.format(len(total_files)))
    out = pool.map(get_result, total_files)
    # export as a csv
    df = pd.DataFrame(out, columns=index + ['accuracy'])
    df[df.path_to_data == './data/u2_raw.jsonl'].to_csv('{}/summary.u2_raw.csv'.format(export_dir))

