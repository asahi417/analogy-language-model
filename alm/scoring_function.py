import logging
from typing import List
from time import time
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

from .lm import TransformersLM
from .data import get_dataset_prompt
from .config_manager import ConfigManager

AGGREGATOR = {'mean': lambda x: sum(x)/len(x), 'max': lambda x: max(x), 'min': lambda x: min(x),
              'none': lambda x: x[0] if len(x) > 0 else 0}


def get_structure(nested_list, batch_size):
    """ create batch while keeping the nested structure """
    batch_data = []
    batch_id = []
    tmp_data_placeholder = []
    tmp_id_placeholder = []
    for n_q, single_q in enumerate(nested_list):
        for n_o, single_option in enumerate(single_q):
            for n_perm_type, pos_neg_permutations in enumerate(single_option):
                for n_perm, single_permutation in enumerate(pos_neg_permutations):
                    tmp_data_placeholder.append(single_permutation)
                    tmp_id_placeholder.append([n_q, n_o, n_perm_type, n_perm])
                    if len(tmp_data_placeholder) == batch_size:
                        batch_data += tmp_data_placeholder
                        batch_id.append(tmp_id_placeholder)
                        tmp_data_placeholder = []
                        tmp_id_placeholder = []
    if len(tmp_data_placeholder) != 0:
        batch_data += tmp_data_placeholder
        batch_id.append(tmp_id_placeholder)
    return batch_data, batch_id


def restore_structure(nested_list, list_score, batch_id):
    """ restore the nested structure from a flatten list """
    list_score = list_score.copy()
    list_placeholder = list(map(
        lambda x: list(map(
            lambda y: (
                [0] * len(nested_list[x[0]][y][0]),
                [0] * len(nested_list[x[0]][y][1])
            ),
            range(len(x[1])))),
        enumerate(nested_list)))

    for single_batch in batch_id:
        for n_q, n_o, n_perm_type, n_perm in single_batch:
            list_placeholder[n_q][n_o][n_perm_type][n_perm] = list_score.pop(0)
    return list_placeholder


class RelationScorer:
    """ Scoring relations with language models """

    def __init__(self,
                 model: str = 'roberta-base',
                 max_length: int = None,
                 cache_dir: str = './cache',
                 num_worker: int = 1):
        """ Scoring relations with language models

        :param model: LM parameter
        :param max_length: LM parameter
        :param cache_dir: LM parameter
        :param num_worker: LM parameter
        """
        logging.info('*** setting up a scorer ***')
        # language model setup
        self.lm = TransformersLM(model=model, max_length=max_length, cache_dir=cache_dir, num_worker=num_worker)
        self.model_name = model

    def analogy_test(self,
                     path_to_data: str,
                     batch_size: int = 4,
                     scoring_method: str = 'ppl',
                     template_types: List = None,
                     permutation_positive: bool = False,
                     permutation_negative: bool = False,
                     aggregation_positive: str = 'mean',
                     aggregation_negative: str = 'none',
                     export_dir: str = './results',
                     debug: bool = False):
        """ relation scoring test on analogy dataset

        :param path_to_data:
        :param scoring_method:
        :param batch_size:
        :param template_types: a list of templates for prompting
        :param permutation_positive: if utilize positive permutation
        :param permutation_negative: if utilize negative permutation
        :param aggregation_positive: aggregation method for positive permutations (`mean`, `max`, `min`)
        :param aggregation_negative: aggregation method for negative permutations (`mean`, `max`, `min`)
        :param export_dir: directory to export the result
        :return:
        """
        start = time()
        # sanity check
        assert ((template_types and len(template_types) == 1) and not permutation_positive) == \
               (aggregation_positive == 'none'), 'permutation/aggregation mismatch (pos)'
        assert permutation_negative == (aggregation_negative != 'none'), 'permutation/aggregation mismatch (neg)'
        assert aggregation_positive in AGGREGATOR.keys()
        assert aggregation_negative in AGGREGATOR.keys()
        aggregator_pos = AGGREGATOR[aggregation_positive]
        aggregator_neg = AGGREGATOR[aggregation_negative]

        # configuration manager
        config = ConfigManager(
            export_dir=export_dir,
            model=self.model_name, max_length=self.lm.max_length, path_to_data=path_to_data,
            scoring_method=scoring_method, template_types=template_types,
            permutation_positive=permutation_positive, permutation_negative=permutation_negative,
            aggregation_positive=aggregation_positive, aggregation_negative=aggregation_negative
        )

        # fetch data
        logging.info('fetch data and templating: {}'.format(path_to_data))
        # get data with all permutation regardless of the configuration
        list_answer, list_nested_sentence, list_stem, list_choice = get_dataset_prompt(path_to_data, template_types)

        # create batch
        logging.info('creating batch (data size: {})'.format(len(list_answer)))
        batch_data, batch_id = get_structure(list_nested_sentence, batch_size)

        # run model prediction over flatten data
        if config.flatten_score is None:
            logging.info('run inference: {}'.format(scoring_method))
            if scoring_method == 'ppl':
                flatten_score = self.lm.get_pseudo_perplexity(batch_data, batch_size=batch_size)
            else:
                raise ValueError('unknown method: {}'.format(scoring_method))
        else:
            flatten_score = config.flatten_score

        # restore the nested structure
        logging.info('restore batch structure')
        score = restore_structure(list_nested_sentence, flatten_score, batch_id)

        if permutation_negative:
            logit_pn = list(map(lambda o: list(map(lambda s: (aggregator_pos(s[0]), aggregator_neg(s[1])), o)), score))
        else:
            logit_pn = list(map(lambda o: list(map(lambda s: (aggregator_pos(s[0]), 0), o)), score))
        logit = list(map(lambda o: list(map(lambda s: s[1] - s[0], o)), logit_pn))
        pred = list(map(lambda x: x.index(max(x)), logit))

        # compute accuracy
        assert len(pred) == len(list_answer)
        accuracy = sum(map(lambda x: int(x[0] == x[1]), zip(pred, list_answer))) / len(list_answer)
        logging.info('accuracy: {}'.format(accuracy))

        # save
        config.save(accuracy=accuracy, flatten_score=flatten_score, logit_pn=logit_pn, logit=logit, prediction=pred)
        logging.info('experiment completed: {} sec in total'.format(time()-start))

