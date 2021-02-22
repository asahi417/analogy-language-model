from itertools import chain
import os
import logging
import requests
import zipfile
import json
from typing import List

__all__ = ('get_dataset_raw', 'AnalogyData')
default_cache_dir_analogy = './data'
root_url_analogy = 'https://github.com/asahi417/AnalogyDataset/raw/master'


def wget(url, cache_dir):
    logging.debug('downloading zip file from {}'.format(url))
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(url)
    with open('{}/{}'.format(cache_dir, filename), "wb") as f:
        r = requests.get(url)
        f.write(r.content)

    with zipfile.ZipFile('{}/{}'.format(cache_dir, filename), 'r') as zip_ref:
        zip_ref.extractall(cache_dir)
    os.remove('{}/{}'.format(cache_dir, filename))


def get_dataset_raw(data_name: str, cache_dir: str = default_cache_dir_analogy):
    """ Get SAT-type dataset: a list of (answer: int, prompts: list, stem: list, choice: list)"""
    assert data_name in ['sat', 'u2', 'u4', 'google', 'bats'], 'unknown data: {}'.format(data_name)
    if not os.path.exists('{}/{}'.format(cache_dir, data_name)):
        url = '{}/{}.zip'.format(root_url_analogy, data_name)
        wget(url, cache_dir)
    with open('{}/{}/test.jsonl'.format(cache_dir, data_name), 'r') as f:
        test = list(filter(None, map(lambda x: json.loads(x) if len(x) > 0 else None, f.read().split('\n'))))
    with open('{}/{}/valid.jsonl'.format(cache_dir, data_name), 'r') as f:
        val = list(filter(None, map(lambda x: json.loads(x) if len(x) > 0 else None, f.read().split('\n'))))
    return val, test


def get_dataset(data: str,
                test_set: bool = True,
                negative_permutation: bool = True,
                marginalize_permutation: bool = False):
    """ Get prompted SAT-type dataset

    :param path_to_data: path to a SAT-type dataset
    :param template_types: a list of templates for prompting, `TEMPLATES`
    :param permutation_negative: if utilize negative permutation
    :param permutation_marginalize: for ppl_pmi marginal likelihood
    :return: a list of (answer: int, prompts: list, stem: list, choice: list)
    """

    def sampling_permutation(a, b, c, d):
        positive = [(a, b, c, d), (a, c, b, d),
                    (b, a, d, c), (b, d, a, c),
                    (c, d, a, b), (c, a, d, b),
                    (d, c, b, a), (d, b, c, a)]
        negative = [(a, b, d, c), (a, c, d, b), (a, d, b, c), (a, d, c, b),
                    (b, a, c, d), (b, c, a, d), (b, c, d, a), (b, d, c, a),
                    (c, a, b, d), (c, b, a, d), (c, b, d, a), (c, d, b, a),
                    (d, a, b, c), (d, a, c, b), (d, b, a, c), (d, c, a, b)]

        if negative_permutation:
            return positive, negative
        else:
            return positive, []

    def single_entry(dictionary):
        a, b = dictionary['stem']

        def perm(c, d):
            return sampling_permutation(a, b, c, d)

        choice = dictionary['choice']
        if marginalize_permutation:
            marginal_choice = list(chain(*[[[i[0], m[1]] for m in choice] for i in choice]))
            permutations = list(map(lambda x: perm(*x), marginal_choice))
        else:
            permutations = list(map(lambda x: perm(*x), dictionary['choice']))
        return dictionary['answer'], permutations

    val, test = get_dataset_raw(data)
    if test_set:
        data = list(map(lambda x: single_entry(x), test))
    else:
        data = list(map(lambda x: single_entry(x), val))
    list_answer = list(list(zip(*data))[0])
    list_nested_permutation = list(list(zip(*data))[1])
    return list_answer, list_nested_permutation


class AnalogyData:

    def __init__(self,
                 data: str,
                 test: bool = False,
                 negative_permutation: bool = True,
                 marginalize_permutation: bool = True):

        self.answer, self.list_nested_permutation = get_dataset(
            data=data,
            test_set=test,
            negative_permutation=negative_permutation,
            marginalize_permutation=marginalize_permutation)
        self.flatten_pos, self.structure_id_pos = self.get_structure(self.list_nested_permutation)
        if negative_permutation:
            self.flatten_neg, self.structure_id_neg = self.get_structure(self.list_nested_permutation, False)
        else:
            self.flatten_neg = self.structure_id_neg = None

    @staticmethod
    def get_structure(nested_list, positive: bool = True):
        """ create batch while keeping the nested structure """
        flatten_data = []
        structure_id = []
        for n_q, single_q in enumerate(nested_list):
            for n_o, single_option in enumerate(single_q):
                perms = single_option[0] if positive else single_option[1]
                for n_perm, single_permutation in enumerate(perms):
                    flatten_data.append(single_permutation)
                    structure_id.append([n_q, n_o, n_perm])
        return flatten_data, structure_id

    def insert_score(self, score_positive: List, score_negative: List = None):
        """ restore the nested structure from a flatten list """
        list_placeholder = list(map(
            lambda x: list(map(
                lambda y: (
                    [0] * len(self.list_nested_permutation[x[0]][y][0]),
                    [0] * len(self.list_nested_permutation[x[0]][y][1])
                ),
                range(len(x[1])))),
            enumerate(self.list_nested_permutation)))
        list_score_pos = score_positive.copy()
        for n_q, n_o, n_perm in self.structure_id_pos:
            list_placeholder[n_q][n_o][0][n_perm] = list_score_pos.pop(0)

        if score_negative is not None and self.structure_id_neg is not None:
            list_score_neg = score_negative.copy()
            for n_q, n_o, n_perm in self.structure_id_neg:
                list_placeholder[n_q][n_o][1][n_perm] = list_score_neg.pop(0)
        return list_placeholder
