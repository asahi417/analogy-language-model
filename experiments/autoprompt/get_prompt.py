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
    word_pairs = word_pairs[:10]
    logging.info('dataset ({}) has {} word pairs'.format(dataset, len(word_pairs)))
    lm = alm.Prompter(model, max_length)
    prompts = lm.replace_mask(word_pairs,
                              seed_type='middle',
                              batch_size=batch)
    assert len(word_pairs) == len(prompts)
    with open('{}/{}.json'.format(export_dit, dataset), 'w') as f:
        json.dump(prompts, f)


if __name__ == '__main__':
    get_prompt('albert-base-v1', 32, 512, 'sat')
    # get_prompt('roberta-large', 12, 512, 'sat')
