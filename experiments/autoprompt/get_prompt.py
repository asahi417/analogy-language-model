import logging
import json
import os
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

import alm
from itertools import chain, product

export_dit = './experiments_results/prompt'
os.makedirs(export_dit, exist_ok=True)
n_blanks = [1, 2, 3, 4, 5]
n_blanks_b = [1, 2, 3]
n_blanks_e = [1, 2, 3]


def get_prompt(model, max_length, batch, dataset, n_blank, n_blank_b, n_blank_e):
    path = '{}/{}.{}.{}.{}.{}.json'.format(export_dit, dataset, model, n_blank, n_blank_b, n_blank_e)

    if os.path.exists(path):
        return
    val, test = alm.get_dataset_raw(dataset)
    word_pairs = list(chain(*[[i['stem']] + i['choice'] for i in val]))
    word_pairs += list(chain(*[[i['stem']] + i['choice'] for i in test]))
    word_pairs_reverse = [[p[1], p[0]] for p in word_pairs]
    word_pairs += word_pairs_reverse
    logging.info('dataset ({}) has {} word pairs'.format(dataset, len(word_pairs)))
    lm = alm.Prompter(model, max_length)
    prompts = lm.generate(word_pairs,
                          n_blank=n_blank,
                          batch_size=batch,
                          debug=True)
    with open(path, 'w') as f:
        json.dump(prompts, f)


if __name__ == '__main__':
    for n_b, n_b_s, n_b_e in product(n_blanks, n_blanks_b, n_blanks_e):
        get_prompt('roberta-large', 32, 512, 'sat', n_b, n_b_s, n_b_e)
