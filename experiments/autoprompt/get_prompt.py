import logging
import json
import os
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

import alm
from itertools import chain

export_dit = './experiments_results/prompt'
os.makedirs(export_dit, exist_ok=True)


def get_prompt(model, max_length, batch, dataset):
    val, test = alm.get_dataset_raw(dataset)
    word_pairs = list(chain(*[[i['stem']] + i['choice'] for i in val]))
    word_pairs += list(chain(*[[i['stem']] + i['choice'] for i in test]))
    print(word_pairs[:10])
    word_pairs_reverse = [[p[1], p[0]] for p in word_pairs]
    print(word_pairs_reverse[:10])
    input()
    word_pairs += word_pairs_reverse
    # word_pairs = [["ewe", "sheep"]]
    logging.info('dataset ({}) has {} word pairs'.format(dataset, len(word_pairs)))
    lm = alm.Prompter(model, max_length)
    prompts = lm.replace_mask(word_pairs,
                              seed_type='middle',
                              batch_size=batch,
                              debug=True)
    with open('{}/{}.{}.json'.format(export_dit, dataset, model), 'w') as f:
        json.dump(prompts, f)


if __name__ == '__main__':
    get_prompt('roberta-large', 32, 512, 'sat')
