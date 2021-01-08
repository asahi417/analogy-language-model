import json
from typing import List
from itertools import permutations, chain

from .prompting_relation import prompting_relation, TEMPLATES

__all__ = 'AnalogyData'


def get_dataset(path_to_data: str):
    """ Get prompted SAT-type dataset: a list of (answer: int, prompts: list, stem: list, choice: list)"""
    with open(path_to_data, 'r') as f:
        return list(filter(None, map(lambda x: json.loads(x) if len(x) > 0 else None, f.read().split('\n'))))


def get_dataset_prompt(path_to_data: str,
                       template_types: List = None,
                       permutation_negative: bool = True,
                       permutation_marginalize: bool = False):
    """ Get prompted SAT-type dataset

    :param path_to_data: path to a SAT-type dataset
    :param template_types: a list of templates for prompting, `TEMPLATES`
    :param permutation_negative: if utilize negative permutation
    :param permutation_marginalize: for ppl_pmi marginal likelihood
    :return: a list of (answer: int, prompts: list, stem: list, choice: list)
    """
    if template_types:
        assert all(t in TEMPLATES.keys() for t in template_types), 'template not found in {}'.format(TEMPLATES.keys())
    else:
        template_types = list(TEMPLATES.keys())

    def sampling_permutation(a, b, c, d):
        all_permutations = list(permutations([a, b, c, d]))
        positive = [(a, b, c, d), (b, a, d, c), (c, d, a, b), (d, c, b, a),
                    (a, c, b, d), (c, a, d, b), (b, d, a, c), (d, b, c, a)]
        if permutation_negative:
            negative = list(filter(lambda x: x not in positive, all_permutations))
        else:
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
            # (prompt, (stem pair, option pair))
            positive_prompt = list(map(lambda x: (x, (a, b, c, d)), positive_prompt))
            negative_prompt = list(map(lambda x: (x, (a, b, c, d)), negative_prompt))
            return positive_prompt, negative_prompt

        choice = dictionary['choice']
        if permutation_marginalize:
            marginal_choice = list(chain(*[[[i[0], m[1]] for m in choice] for i in choice]))
            prompts = list(map(lambda x: single_prompt(*x), marginal_choice))
        else:
            prompts = list(map(lambda x: single_prompt(*x), dictionary['choice']))
        return dictionary['answer'], prompts

    data = list(map(lambda x: single_entry(x), get_dataset(path_to_data)))
    list_answer = list(list(zip(*data))[0])
    list_nested_sentence = list(list(zip(*data))[1])
    return list_answer, list_nested_sentence


class AnalogyData:

    def __init__(self,
                 path_to_data: str,
                 template_types: List = None,
                 permutation_negative: bool = True,
                 permutation_marginalize: bool = True):

        self.answer, self.list_nested_sentence = get_dataset_prompt(
            path_to_data=path_to_data,
            template_types=template_types,
            permutation_negative=permutation_negative,
            permutation_marginalize=permutation_marginalize)
        self.flatten_prompt_pos, self.structure_id_pos = self.get_structure(self.list_nested_sentence)
        if permutation_negative:
            self.flatten_prompt_neg, self.structure_id_neg = self.get_structure(self.list_nested_sentence, False)
        else:
            self.flatten_prompt_neg = self.structure_id_neg = None

    def get_prompt(self,
                   return_relation_pairs: bool = True,
                   positive: bool = True):
        if positive:
            prompt = self.flatten_prompt_pos
        else:
            prompt = self.flatten_prompt_neg
        if not prompt:
            return None, None
        prompt_list = list(list(zip(*prompt))[0])
        if return_relation_pairs:
            relation_list = list(list(zip(*prompt))[1])
            return prompt_list, relation_list
        else:
            return prompt_list,

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
        print('pos: {}'.format(len(score_positive)))
        print('neg: {}'.format(len(score_negative)))
        list_placeholder = list(map(
            lambda x: list(map(
                lambda y: (
                    [0] * len(self.list_nested_sentence[x[0]][y][0]),
                    [0] * len(self.list_nested_sentence[x[0]][y][1])
                ),
                range(len(x[1])))),
            enumerate(self.list_nested_sentence)))

        list_score_pos = score_positive.copy()

        for n_q, n_o, n_perm in self.structure_id_pos:
            list_placeholder[n_q][n_o][0][n_perm] = list_score_pos.pop(0)

        if score_negative is not None and self.structure_id_neg is not None:
            list_score_neg = score_negative.copy()
            for n_q, n_o, n_perm in self.structure_id_neg:
                list_placeholder[n_q][n_o][1][n_perm] = list_score_neg.pop(0)
        print('pos: {}'.format(len(score_positive)))
        print('neg: {}'.format(len(score_negative)))
        return list_placeholder
