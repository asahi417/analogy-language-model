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
    ['she-to-as'],  # u2 second
    ['as-what-same']  # u4 second, sat first
]
# u4 ('p_0', 'min'), u2 ('p_4', 'p_0'), sat ('p_2', 'min')
aggregation_positive = ['min', 'p_0', 'p_4', 'p_2']
aggregation_negative = ['max', 'mean', 'min', 'p_0', 'p_1', 'p_2', 'p_3', 'p_4', 'p_5', 'p_6', 'p_7',
                        'p_8', 'p_9', 'p_10', 'p_11']
ppl_pmi_aggregation = ['max', 'mean', 'min', 'p_0', 'p_1']
lambdas = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
alphas = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
permutation_negative_weight = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
index = ['model', 'path_to_data', 'scoring_method', 'template_types', 'aggregation_positive',
         'aggregation_negative', 'ppl_pmi_lambda', 'ppl_pmi_alpha', 'ppl_pmi_aggregation',
         'permutation_negative_weight']
total_files = glob('./{}/outputs/*'.format(export_dir))
pbar = tqdm.tqdm(total=int(len(total_files)/os.cpu_count()))


def get_options():
    parser = argparse.ArgumentParser(description='command line tool to test finetuned NER model',)
    parser.add_argument('-e', '--experiment', default=None, type=str)

    return parser.parse_args()


def main(path_to_data):
    # get accuracy
    for template in templates:
        scorer = alm.RelationScorer(model='roberta-large', max_length=32)
        scorer.analogy_test(
            skip_duplication_check=False,
            scoring_method='ppl_pmi',
            path_to_data=path_to_data,
            template_types=template,
            aggregation_positive=aggregation_positive,
            permutation_negative=True,
            aggregation_negative=aggregation_negative,
            ppl_pmi_lambda=lambdas,
            ppl_pmi_alpha=alphas,
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
    if opt.experiment == 'sat_package_v3':
        main(path_to_data='./data/sat_package_v3.jsonl')
    elif opt.experiment == 'u2_raw':
        main(path_to_data='./data/u2_raw.jsonl')
    elif opt.experiment == 'u4_raw':
        main(path_to_data='./data/u4_raw.jsonl')
    else:
        print('CPU count: {}'.format(os.cpu_count()))
        pool = Pool()  # Create a multiprocessing Pool
        total_files = glob('./{}/outputs/*'.format(export_dir))
        print('total file: {}'.format(len(total_files)))

        out = pool.map(get_result, total_files)
        # export as a csv
        df = pd.DataFrame(out, columns=index + ['accuracy'])
        df[df.path_to_data == './data/sat_package_v3.jsonl'].to_csv('{}/summary.sat_package_v3.csv'.format(export_dir))
        df[df.path_to_data == './data/u2_raw.jsonl'].to_csv('{}/summary.u2_raw.csv'.format(export_dir))
        df[df.path_to_data == './data/u4_raw.jsonl'].to_csv('{}/summary.u4_raw.csv'.format(export_dir))

