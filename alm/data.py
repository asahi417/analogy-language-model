import json
import os
from typing import List
from itertools import permutations, chain

from .prompting_relation import prompting_relation, TEMPLATES

__all__ = ('get_dataset', 'get_dataset_prompt')


def get_dataset(path_to_data: str):
    """ get prompted SAT-type dataset: a list of (answer: int, prompts: list, stem: list, choice: list)"""
    assert os.path.exists(path_to_data)

    with open(path_to_data, 'r') as f:
        return list(filter(None, map(lambda x: json.loads(x) if len(x) > 0 else None, f.read().split('\n'))))


def get_dataset_prompt(path_to_data: str,
                       template_types: List = None,
                       permutation_positive: bool = True,
                       permutation_negative: bool = True):
    """ get prompted SAT-type dataset

    :param path_to_data:
    :param template_types: a list of templates for prompting
    :param permutation_positive: if utilize positive permutation
    :param permutation_negative: if utilize negative permutation
    :return: a list of (answer: int, prompts: list, stem: list, choice: list)
    """
    if template_types:
        assert all(t in TEMPLATES.keys() for t in template_types), 'template not found in {}'.format(TEMPLATES.keys())
    else:
        template_types = list(TEMPLATES.keys())

    def sampling_permutation(a, b, c, d):
        all_permutations = list(permutations([a, b, c, d]))
        positive = [(a, b, c, d), (b, a, d, c), (c, d, a, b), (d, c, b, a),
                    (a, d, c, b), (d, a, b, c), (c, b, a, d), (b, c, d, a)]
        negative = list(filter(lambda x: x not in positive, all_permutations))
        if not permutation_positive:
            positive = [(a, b, c, d)]
        if not permutation_negative:
            negative = []
        return positive, negative

    def single_entry(dictionary):
        a = dictionary['stem'][0]
        b = dictionary['stem'][1]

        def single_prompt(c, d):
            positive, negative = sampling_permutation(a, b, c, d)
            positive_prompt = list(chain(*map(
                lambda x: list(map(
                    lambda t: prompting_relation(
                        subject_stem=x[0], object_stem=x[1], subject_analogy=x[2], object_analogy=x[3],
                        template_type=t),
                    template_types)),
                positive)))
            negative_prompt = list(chain(*map(
                lambda x: list(map(
                    lambda t: prompting_relation(
                        subject_stem=x[0], object_stem=x[1], subject_analogy=x[2], object_analogy=x[3],
                        template_type=t),
                    template_types)),
                negative)))
            return positive_prompt, negative_prompt

        prompts = list(map(lambda x: single_prompt(*x), dictionary['choice']))
        return dictionary['answer'], prompts, dictionary['stem'], dictionary['choice']

    data = list(map(lambda x: single_entry(x), get_dataset(path_to_data)))
    list_answer = list(list(zip(*data))[0])
    list_nested_sentence = list(list(zip(*data))[1])
    list_stem = list(list(zip(*data))[2])
    list_choice = list(list(zip(*data))[3])
    return list_answer, list_nested_sentence, list_stem, list_choice
