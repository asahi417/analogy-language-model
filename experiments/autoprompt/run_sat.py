import logging
import json
import pickle
import os
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

import alm
from itertools import chain, product

export_dit = './experiments_results/prompt'
seed_types = ['middle', 'whole']
n_blanks = [4, 5, 6]
dataset = 'sat'
model = 'roberta-large'
batch = 512
max_length = 24


val, test = alm.get_dataset_raw(dataset)
full_data = val + test
lm = alm.Prompter(model, max_length)


def get_partition(_list):
    length = list(map(lambda x: len(x), _list))
    return list(map(lambda x: [sum(length[:x]), sum(length[:x + 1])], range(len(length))))


def main(n_blank, seed_type):
    output_file = '{}/{}.{}.{}.{}.pkl'.format(export_dit, dataset, model, n_blank, seed_type)
    if os.path.exists(output_file):
        with open(output_file, "rb") as fp:
            score = pickle.load(fp)
        list_answer = [data['answer'] for data in full_data]
    else:
        dict_file = '{}/{}.{}.{}.{}.no_rep.json'.format(export_dit, dataset, model, n_blank, seed_type)
        if not os.path.exists(dict_file):
            return

        with open(dict_file, 'r') as f:
            prompt_dict = json.load(f)
        list_answer = []
        list_prompt = []
        for data in full_data:
            list_answer.append(data['answer'])
            h, t = data['stem']
            template = prompt_dict['||'.join([h, t])]
            list_prompt = [template.copy().replace(h, h_c).replace(t, t_c) for h_c, t_c in data['choice']]
        partition = get_partition(list_prompt)
        score = lm.get_perplexity(list(chain(*list_prompt)))
        score = [score[s:e] for s, e in partition]
        with open(output_file, 'wb') as fp:
            pickle.dump(score, fp)
    accuracy = []
    for a, s in zip(list_answer, score):
        p = s.index(min(s))
        accuracy.append(int(a == p))
    accuracy = sum(accuracy)/len(accuracy)
    return accuracy


if __name__ == '__main__':
    for _s, b in product(seed_types, n_blanks):
        acc = main(b, _s)
        print('\nseed: {}, blank: {}'.format(_s, b))
        print(acc)

