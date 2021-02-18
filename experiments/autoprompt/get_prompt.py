import logging
import json
import os
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

import alm
from itertools import chain, product

export_dit = './experiments_results/prompt'
os.makedirs(export_dit, exist_ok=True)
seed_types = ['middle', 'whole']
n_blanks = [5, 6, 7]


def get_prompt(model, max_length, batch, dataset, n_blank, seed_type):
    val, test = alm.get_dataset_raw(dataset)
    word_pairs = list(chain(*[[i['stem']] + i['choice'] for i in val]))
    word_pairs += list(chain(*[[i['stem']] + i['choice'] for i in test]))
    word_pairs_reverse = [[p[1], p[0]] for p in word_pairs]
    word_pairs += word_pairs_reverse
    logging.info('dataset ({}) has {} word pairs'.format(dataset, len(word_pairs)))
    lm = alm.Prompter(model, max_length)
    prompts = lm.replace_mask(word_pairs,
                              seed_type=seed_type,
                              n_blank=n_blank,
                              batch_size=batch,
                              debug=True)
    with open('{}/{}.{}.{}.{}.json'.format(export_dit, dataset, model, n_blank, seed_type), 'w') as f:
        json.dump(prompts, f)


if __name__ == '__main__':
    for s, b in product(seed_types, n_blanks):
        try:
            get_prompt('roberta-large', 32, 512, 'sat', b, s)
        except Exception:
            print('\n\n\nERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR\n\n\n')
            pass
