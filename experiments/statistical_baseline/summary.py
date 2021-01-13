import json
from glob import glob
from typing import List

import pandas as pd


def cos_similarity(a_: List, b: List):
    assert len(a_) == len(b)
    norm_a = sum(map(lambda x: x * x, a_)) ** 0.5
    norm_b = sum(map(lambda x: x * x, b)) ** 0.5
    inner_prod = sum(map(lambda x: x[0] * x[1], zip(a_, b)))
    return inner_prod / (norm_a * norm_b)


def get_dataset(path_to_data: str):
    """ Get prompted SAT-type dataset: a list of (answer: int, prompts: list, stem: list, choice: list)"""
    with open(path_to_data, 'r') as f:
        return list(filter(None, map(lambda x: json.loads(x) if len(x) > 0 else None, f.read().split('\n'))))


def get_pmi(smooth, data):
    all_dict = {}
    for i in glob('./cache/ppmi/{}*smooth={}.csv'.format(data, smooth)):
        with open(i, 'r') as f:
            tmp = list(filter(lambda x: len(x) == 3, map(lambda x: x.split('\t'), f.read().split('\n'))))
        tmp_dict = {'{}-{}'.format(h, t): float(score) for h, t, score in tmp if score != 'x'}
        if len(set(tmp_dict.keys()).intersection(set(all_dict.keys()))) != 0:
            inter = set(tmp_dict.keys()).intersection(set(all_dict.keys()))
            for ii in inter:
                assert all_dict[ii] == tmp_dict[ii]
        all_dict.update(tmp_dict)
    return all_dict


def get_accuracy_pmi(dataset, pmi, none_index_past):
    none_index_past = none_index_past.copy()
    accuracy = []
    none_index = []
    for n, i in enumerate(dataset):
        if n in none_index_past:
            continue
        a_ = i['answer']
        score = [pmi['-'.join(c)] if '-'.join(c) in pmi.keys() else -100 for c in i['choice']]
        if all(s == -100 for s in score):
            none_index.append(n)
        else:
            accuracy.append(int(score.index(max(score)) == a_))
    accuracy = sum(accuracy)/len(accuracy)
    none_index = none_index + none_index_past
    print('invalid data: {}'.format(len(none_index)))
    print('accuracy: {}'.format(accuracy))
    return accuracy, none_index


def get_accuracy_rel(dataset, relative):
    accuracy = []
    none_index = []
    for n, i in enumerate(dataset):
        stem = '-'.join(i['stem'])
        if stem not in relative.keys():
            none_index.append(n)
        else:
            a_ = i['answer']
            e_stem = relative[stem]
            score = [
                cos_similarity(relative['-'.join(c)], e_stem) if '-'.join(c) in relative.keys() else -100 for c in i['choice']]
            if all(s == -100 for s in score):
                none_index.append(n)
            else:
                accuracy.append(int(score.index(max(score)) == a_))
    accuracy = sum(accuracy)/len(accuracy)
    print('invalid data: {}'.format(len(none_index)))
    print('accuracy: {}'.format(accuracy))
    return accuracy, none_index


def load_relative(path_to_relative):
    with open(path_to_relative, 'r') as f:
        tmp = [v.split(' ') for v in f.read().split('\n')[1:]]
        relative_dict = {v[0].replace('__', '-'): list(map(float, v[1:])) for v in tmp if len(v[0]) != 0}
        return relative_dict


if __name__ == '__main__':
    result = {}
    
    for r, path in zip(
            ['./cache/relative/relative_sat.txt', './cache/relative/relative_u2.txt', './cache/relative/relative_u4.txt'],
            ['./data/sat_package_v3.jsonl', './data/u2_raw.jsonl', './data/u4_raw.jsonl']):
        embeddings = load_relative(r)
        if 'sat' in path:
            pmi_dict_0 = get_pmi(0.5, 'sat')
            pmi_dict_1 = get_pmi(1, 'sat')
        else:
            pmi_dict_0 = get_pmi(0.5, 'u')
            pmi_dict_1 = get_pmi(1, 'u')
        valid_k = list(set(pmi_dict_0.keys()).intersection(set(embeddings.keys())))
        print(len(valid_k), len(pmi_dict_0.keys()), len(embeddings.keys()))

        embeddings = {k: v for k, v in embeddings.items() if k in valid_k}
        pmi_dict_0 = {k: v for k, v in pmi_dict_0.items() if k in valid_k}
        pmi_dict_1 = {k: v for k, v in pmi_dict_1.items() if k in valid_k}
        print(len(embeddings), len(pmi_dict_0), len(pmi_dict_1))

        data = get_dataset(path)

        # get none intersection
        a, none_ind = get_accuracy_rel(data, embeddings)
        result[len(result)] = {'smooth': None, 'accuracy': a, 'path_to_data': path, 'method': 'relation_embedding'}

        a, none_ind_ = get_accuracy_pmi(data, pmi_dict_0, none_ind)
        assert none_ind == none_ind_
        result[len(result)] = {'smooth': 0.5, 'accuracy': a, 'path_to_data': path, 'method': 'pmi'}

        a, none_ind_ = get_accuracy_pmi(data, pmi_dict_1, none_ind)
        assert none_ind == none_ind_
        result[len(result)] = {'smooth': 1, 'accuracy': a, 'path_to_data': path, 'method': 'pmi'}

    df = pd.DataFrame(result).T
    print(df)
