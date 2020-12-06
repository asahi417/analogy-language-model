import logging
from typing import List
from time import time
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

from . import TransformersLM
from .data import get_dataset_prompt
from .config_manager import ConfigManager

AGGREGATOR = {
    'mean': lambda x: sum(x)/len(x), 'max': lambda x: max(x), 'min': lambda x: min(x),
    'p_1': lambda x: x[1], 'p_2': lambda x: x[2], 'p_3': lambda x: x[3],
    'p_4': lambda x: x[4], 'p_5': lambda x: x[5], 'p_6': lambda x: x[6], 'p_7': lambda x: x[7],
    'p_0': lambda x: x[0], 'none': lambda x: 0
}


def get_structure(nested_list, batch_size, positive: bool = True):
    """ create batch while keeping the nested structure """
    batch_data = []
    batch_id = []
    tmp_id_placeholder = []
    tmp_data_placeholder = []
    for n_q, single_q in enumerate(nested_list):
        for n_o, single_option in enumerate(single_q):
            permutations = single_option[0] if positive else single_option[1]
            for n_perm, single_permutation in enumerate(permutations):
                tmp_data_placeholder.append(single_permutation)
                tmp_id_placeholder.append([n_q, n_o, n_perm])
                if len(tmp_id_placeholder) == batch_size:
                    batch_data += tmp_data_placeholder
                    batch_id.append(tmp_id_placeholder)
                    tmp_data_placeholder = []
                    tmp_id_placeholder = []

    if len(tmp_data_placeholder) != 0:
        batch_data += tmp_data_placeholder
        batch_id.append(tmp_id_placeholder)
    return batch_data, batch_id


# def get_structure(nested_list, batch_size):
#     """ create batch while keeping the nested structure """
#     batch_data_pos = []
#     batch_id_pos = []
#     batch_data_neg = []
#     batch_id_neg = []
#     tmp_data_placeholder_neg = []
#     tmp_id_placeholder_neg = []
#     tmp_id_placeholder_pos = []
#     tmp_data_placeholder_pos = []
#     for n_q, single_q in enumerate(nested_list):
#         for n_o, single_option in enumerate(single_q):
#             # positive
#             for n_perm, single_permutation in enumerate(single_option[0]):
#                 tmp_data_placeholder_pos.append(single_permutation)
#                 tmp_id_placeholder_pos.append([n_q, n_o, n_perm])
#                 if len(tmp_id_placeholder_pos) == batch_size:
#                     batch_data_pos += tmp_data_placeholder_pos
#                     batch_id_pos.append(tmp_id_placeholder_pos)
#                     tmp_data_placeholder_pos = []
#                     tmp_id_placeholder_pos = []
#             # negative
#             for n_perm, single_permutation in enumerate(single_option[1]):
#                 tmp_data_placeholder_neg.append(single_permutation)
#                 tmp_id_placeholder_neg.append([n_q, n_o, n_perm])
#                 if len(tmp_id_placeholder_neg) == batch_size:
#                     batch_data_neg += tmp_data_placeholder_neg
#                     batch_id_neg.append(tmp_id_placeholder_neg)
#                     tmp_data_placeholder_neg = []
#                     tmp_id_placeholder_neg = []
#
#     if len(tmp_data_placeholder_pos) != 0:
#         batch_data_pos += tmp_data_placeholder_pos
#         batch_id_pos.append(tmp_id_placeholder_pos)
#     if len(tmp_data_placeholder_neg) != 0:
#         batch_data_neg += tmp_data_placeholder_neg
#         batch_id_neg.append(tmp_id_placeholder_neg)
#     return (batch_data_pos, batch_data_neg), (batch_id_pos, batch_id_neg)


def restore_structure(nested_list, list_score_pos, batch_id_pos, list_score_neg=None, batch_id_neg=None):
    """ restore the nested structure from a flatten list """
    list_placeholder = list(map(
        lambda x: list(map(
            lambda y: (
                [0] * len(nested_list[x[0]][y][0]),
                [0] * len(nested_list[x[0]][y][1])
            ),
            range(len(x[1])))),
        enumerate(nested_list)))

    list_score_pos = list_score_pos.copy()

    for single_batch in batch_id_pos:
        for n_q, n_o, n_perm in single_batch:
            list_placeholder[n_q][n_o][0][n_perm] = list_score_pos.pop(0)
    if list_score_neg is not None and batch_id_neg is not None:
        list_score_neg = list_score_neg.copy()
        for single_batch in batch_id_neg:
            for n_q, n_o, n_perm in single_batch:
                list_placeholder[n_q][n_o][1][n_perm] = list_score_neg.pop(0)
    return list_placeholder


class RelationScorer:
    """ Scoring relations with language models """

    def __init__(self,
                 model: str = 'roberta-base',
                 max_length: int = None,
                 cache_dir: str = './cache',
                 num_worker: int = 1,
                 embedding_mode: bool = False):
        """ Scoring relations with language models

        :param model: LM parameter
        :param max_length: LM parameter
        :param cache_dir: LM parameter
        :param num_worker: LM parameter
        """
        logging.info('*** setting up a scorer ***')
        # language model setup
        self.lm = TransformersLM(
            model=model,
            max_length=max_length,
            cache_dir=cache_dir,
            num_worker=num_worker,
            embedding_mode=embedding_mode)
        self.model_name = model

    def analogy_test(self,
                     path_to_data: str,
                     batch_size: int = 4,
                     scoring_method: str = 'ppl',
                     template_types: List = None,
                     permutation_negative: bool = False,
                     aggregation_positive: str = 'mean',
                     aggregation_negative: str = 'none',
                     skip_scoring_prediction: bool = False,
                     export_dir: str = './results'):
        """ relation scoring test on analogy dataset

        :param path_to_data:
        :param scoring_method:
        :param batch_size:
        :param template_types: a list of templates for prompting
        :param permutation_negative: if utilize negative permutation
        :param aggregation_positive: aggregation method for positive permutations (`mean`, `max`, `min`)
        :param aggregation_negative: aggregation method for negative permutations (`mean`, `max`, `min`)
        :param skip_scoring_prediction:
        :param export_dir: directory to export the result
        :return:
        """
        start = time()
        # sanity check
        assert permutation_negative == (aggregation_negative != 'none'), 'permutation/aggregation mismatch (neg)'
        assert aggregation_positive in AGGREGATOR.keys()
        assert aggregation_negative in AGGREGATOR.keys()
        aggregator_pos = AGGREGATOR[aggregation_positive]
        aggregator_neg = AGGREGATOR[aggregation_negative]

        # configuration manager
        config = ConfigManager(
            export_dir=export_dir,
            model=self.model_name, max_length=self.lm.max_length, path_to_data=path_to_data,
            scoring_method=scoring_method, template_types=template_types, permutation_negative=permutation_negative,
            aggregation_positive=aggregation_positive, aggregation_negative=aggregation_negative
        )

        # fetch data
        logging.info('fetch data and templating: {}'.format(path_to_data))
        # get data with all permutation regardless of the configuration
        list_answer, list_nested_sentence, list_stem, list_choice = get_dataset_prompt(
            path_to_data, template_types, permutation_negative=permutation_negative)

        # create batch
        logging.info('creating batch (data size: {})'.format(len(list_answer)))
        batch_data_pos, batch_id_pos = get_structure(list_nested_sentence, batch_size, positive=True)
        if permutation_negative:
            batch_data_neg, batch_id_neg = get_structure(list_nested_sentence, batch_size, positive=False)
        else:
            batch_data_neg, batch_id_neg = None, None

        def prediction(batch_data, cached_score):
            if batch_data is None:
                logging.info(' * skip permutation')
                return None
            if cached_score:
                logging.info(' * load score')
                return cached_score
            logging.info(' * run inference')
            if scoring_method == 'ppl':
                return self.lm.get_perplexity(batch_data, batch_size=batch_size)
            else:
                raise ValueError('unknown method: {}'.format(scoring_method))

        # positive permutation
        logging.info('positive permutation')
        score_pos = prediction(batch_data_pos, config.flatten_score_positive)
        logging.info('negative permutation')
        score_neg = prediction(batch_data_neg, config.flatten_score_negative)

        config.cache_scores(flatten_score_positive=score_pos, flatten_score_negative=score_neg)

        if skip_scoring_prediction:
            return

        # restore the nested structure
        logging.info('restore batch structure')
        score = restore_structure(list_nested_sentence, score_pos, batch_id_pos, score_neg, batch_id_neg)
        logit_pn = list(map(lambda o: list(map(lambda s: (aggregator_pos(s[0]), aggregator_neg(s[1])), o)), score))
        logit = list(map(lambda o: list(map(lambda s: s[1] - s[0], o)), logit_pn))
        pred = list(map(lambda x: x.index(max(x)), logit))

        # compute accuracy
        assert len(pred) == len(list_answer)
        accuracy = sum(map(lambda x: int(x[0] == x[1]), zip(pred, list_answer))) / len(list_answer)
        logging.info('accuracy: {}'.format(accuracy))

        # save
        config.save(accuracy=accuracy, logit_pn=logit_pn, logit=logit, prediction=pred)
        logging.info('experiment completed: {} sec in total'.format(time()-start))

