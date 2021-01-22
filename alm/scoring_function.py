import logging
from typing import List
from time import time
from itertools import permutations, product
from math import log
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

import torch
from . import TransformersLM
from .data import AnalogyData
from .config_manager import ConfigManager

AGGREGATOR = {
    'mean': lambda x: sum(x)/len(x), 'max': lambda x: max(x), 'min': lambda x: min(x),
    'index_0': lambda x: x[0], 'index_1': lambda x: x[1], 'index_2': lambda x: x[2], 'index_3': lambda x: x[3],
    'index_4': lambda x: x[4], 'index_5': lambda x: x[5], 'index_6': lambda x: x[6], 'index_7': lambda x: x[7],
    'index_8': lambda x: x[8], 'index_9': lambda x: x[9], 'index_10': lambda x: x[10], 'index_11': lambda x: x[11],
    'index_12': lambda x: x[12], 'index_13': lambda x: x[13], 'index_14': lambda x: x[14], 'index_15': lambda x: x[15],
    'none': lambda x: 0
}


class RelationScorer:
    """ Scoring relations with language models """

    def __init__(self,
                 model: str = 'roberta-base',
                 max_length: int = 32,
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

    def release_cache(self):
        if self.lm.device == "cuda":
            torch.cuda.empty_cache()

    def analogy_test(self,
                     data: str,
                     template_type: str,
                     test: bool = False,
                     batch_size: int = 4,
                     scoring_method: str = 'ppl',
                     positive_permutation_aggregation: (str, List) = 'mean',
                     negative_permutation: bool = False,
                     negative_permutation_weight: (float, List) = 1.0,
                     negative_permutation_aggregation: (str, List) = 'none',
                     pmi_aggregation: str = None,
                     pmi_lambda: float = 1.0,
                     ppl_pmi_aggregation: (str, List) = None,
                     ppl_pmi_lambda: (float, List) = 1.0,
                     ppl_pmi_alpha: (float, List) = 1.0,
                     skip_scoring_prediction: bool = False,
                     export_dir: str = './experiments_results',
                     no_inference: bool = False,
                     overwrite_output: bool = False):
        """ relation scoring test on analogy dataset

        :param path_to_data:
        :param scoring_method:
        :param ppl_pmi_aggregation: upto p_1
        :param pmi_aggregation: upto p_11
        :param aggregation_positive: upto p_7
        :param aggregation_negative: upto p_15
        :param pmi_lambda:
        :param ppl_pmi_lambda:
        :param batch_size:
        :param template_types: a list of templates for prompting
        :param permutation_negative: if utilize negative permutation
        :param skip_scoring_prediction:
        :param no_inference: use only cached score
        :param export_dir: directory to export the result
        :param overwrite_output: overwrite existing result file
        :return:
        """
        # start = time()
        #

        ##############
        # get scores #
        ##############
        score_pos, score_neg, config, data_instance = self.get_score(
            export_dir='{}/logit'.format(export_dir),
            test=test,
            template_type=template_type,
            data=data,
            batch_size=batch_size,
            scoring_method=scoring_method,
            pmi_lambda=pmi_lambda,
            negative_permutation=negative_permutation,
            no_inference=no_inference
        )
        if skip_scoring_prediction:
            return
        #
        # ##################
        # # restore scores #
        # ##################
        # # negative pmi score aggregation
        # if scoring_method == 'pmi':
        #     assert pmi_aggregation, 'undefined pmi aggregation'
        #     # we use same aggregation method for both positive/negative permutations
        #     logging.info("PMI aggregator: {}".format(pmi_aggregation))
        #     aggregator = AGGREGATOR[pmi_aggregation]
        #     score_pos = list(map(lambda x: aggregator(x), score_pos))
        #     if score_neg:
        #         score_neg = list(map(lambda x: aggregator(x), score_neg))
        # else:
        #     assert pmi_lambda == 1.0 and pmi_aggregation is None, 'pmi_lambda/pmi_aggregation should be default'
        #
        # # restore the nested structure
        # logging.info('restore batch structure')
        # score = data_instance.insert_score(score_pos, score_neg)
        #
        # ##############
        # # get output #
        # ##############
        # # get all outputs from any combinations
        # if type(ppl_pmi_aggregation) is not list:
        #     ppl_pmi_aggregation = [ppl_pmi_aggregation]
        # if type(ppl_pmi_lambda) is not list:
        #     ppl_pmi_lambda = [ppl_pmi_lambda]
        # if type(ppl_pmi_alpha) is not list:
        #     ppl_pmi_alpha = [ppl_pmi_alpha]
        # if type(aggregation_positive) is not list:
        #     aggregation_positive = [aggregation_positive]
        # if type(aggregation_negative) is not list:
        #     aggregation_negative = [aggregation_negative]
        # if type(permutation_negative_weight) is not list:
        #     permutation_negative_weight = [permutation_negative_weight]
        # all_config = list(
        #     product(ppl_pmi_aggregation, ppl_pmi_lambda, ppl_pmi_alpha, aggregation_positive, aggregation_negative,
        #             permutation_negative_weight))
        #
        # def get_config(_ppl_pmi_aggregation, _ppl_pmi_lambda, _ppl_pmi_alpha, _aggregation_positive,
        #                _aggregation_negative, _permutation_negative_weight):
        #     # configuration manager
        #     return ConfigManager(
        #         skip_duplication_check=skip_duplication_check,
        #         skip_flatten_score=True,
        #         export_dir=export_dir,
        #         pmi_aggregation=pmi_aggregation,
        #         ppl_pmi_aggregation=_ppl_pmi_aggregation,
        #         pmi_lambda=pmi_lambda,
        #         model=self.model_name,
        #         max_length=self.lm.max_length,
        #         path_to_data=path_to_data,
        #         scoring_method=scoring_method,
        #         template_types=template_types,
        #         permutation_negative=permutation_negative,
        #         aggregation_positive=_aggregation_positive,
        #         aggregation_negative=_aggregation_negative,
        #         ppl_pmi_lambda=_ppl_pmi_lambda,
        #         ppl_pmi_alpha=_ppl_pmi_alpha,
        #         permutation_negative_weight=_permutation_negative_weight
        #     )
        #
        # logging.info('aggregate configuration: {}'.format(len(all_config)))
        # all_config = list(map(lambda c: get_config(*c), all_config))
        # if not overwrite_output:
        #     all_config = list(filter(lambda c: not c.output_exist, all_config))
        #     logging.info('remaining configurations: {}'.format(len(all_config)))
        #
        # for n, config in enumerate(all_config):
        #     logging.info('##### CONFIG {}/{} #####'.format(n, len(all_config)))
        #     assert config.config['aggregation_positive'] in AGGREGATOR.keys()
        #     assert config.config['aggregation_negative'] in AGGREGATOR.keys()
        #     aggregator_pos = AGGREGATOR[config.config['aggregation_positive']]
        #     aggregator_neg = AGGREGATOR[config.config['aggregation_negative']]
        #     if not overwrite_output:
        #         assert not config.output_exist
        #
        #     # ppl_pmi aggregation
        #     if scoring_method == 'ppl_pmi':
        #         # TODO: validate on multiple templates
        #         assert config.config['ppl_pmi_aggregation'] is not None
        #
        #         aggregator = AGGREGATOR[config.config['ppl_pmi_aggregation']]
        #
        #         def compute_pmi(ppl_scores):
        #             opt_length = len(ppl_scores) ** 0.5
        #             assert opt_length.is_integer(), 'something wrong'
        #             opt_length = int(opt_length)
        #
        #             if all(s == 0 for s in ppl_scores):
        #                 return [0] * opt_length
        #
        #             # conditional negative log likelihood (fixed head and tail tokens)
        #             ppl_in_option = list(map(lambda x: ppl_scores[opt_length * x + x], range(opt_length)))
        #             negative_log_likelihood_cond = list(map(lambda x: log(x / sum(ppl_in_option)), ppl_in_option))
        #
        #             # marginal negative log likelihood (tail token)
        #             ppl_out_option = list(map(
        #                 lambda x: sum(map(lambda y: ppl_scores[x + opt_length * y], range(opt_length))),
        #                 range(opt_length)))
        #             negative_log_likelihood_mar_t = list(map(lambda x: log(x / sum(ppl_out_option)), ppl_out_option))
        #
        #             # marginal negative log likelihood (head token)
        #             ppl_out_option = list(map(
        #                 lambda x: sum(ppl_scores[x * opt_length: (x + 1) * opt_length]),
        #                 range(opt_length)))
        #             negative_log_likelihood_mar_h = list(map(lambda x: log(x / sum(ppl_out_option)), ppl_out_option))
        #
        #             # negative pmi approx by perplexity difference: higher is better
        #             neg_pmi = list(map(
        #                 lambda x: x[0] * config.config['ppl_pmi_lambda'] - aggregator([x[1], x[2]]) * config.config['ppl_pmi_alpha'],
        #                 zip(negative_log_likelihood_cond, negative_log_likelihood_mar_h, negative_log_likelihood_mar_t)))
        #             return neg_pmi
        #
        #         pmi = list(map(lambda o: (
        #             list(map(lambda x: compute_pmi(list(map(lambda s: s[0][x], o))), range(8))),
        #             list(map(lambda x: compute_pmi(list(map(lambda s: s[1][x] if len(s[1]) != 0 else 0, o))), range(16)))
        #         ), score))
        #
        #         logit_pn = list(map(
        #             lambda s: (
        #                 list(zip(
        #                     list(map(lambda o: aggregator_pos(o), list(zip(*s[0])))),
        #                     list(map(lambda o: aggregator_neg(o), list(zip(*s[1]))))
        #                 ))
        #             ),
        #             pmi))
        #
        #     else:
        #         logit_pn = list(map(
        #             lambda o: list(map(
        #                 lambda s: (aggregator_pos(s[0]), aggregator_neg(s[1])),
        #                 o)),
        #             score))
        #     permutation_negative_weight = 1 if config.config['permutation_negative_weight'] is None else config.config['permutation_negative_weight']
        #     logit = list(map(lambda o: list(map(lambda s: permutation_negative_weight * s[1] - s[0], o)), logit_pn))
        #     pred = list(map(lambda x: x.index(max(x)), logit))
        #
        #     # compute accuracy
        #     assert len(pred) == len(data_instance.answer)
        #     accuracy = sum(map(lambda x: int(x[0] == x[1]), zip(pred, data_instance.answer))) / len(data_instance.answer)
        #     logging.info('accuracy: {}'.format(accuracy))
        #     config.save(accuracy=accuracy, logit_pn=logit_pn, logit=logit, prediction=pred)
        # logging.info('experiment completed: {} sec in total'.format(time()-start))

    def get_score(self,
                  export_dir,
                  test,
                  template_type,
                  data,
                  batch_size,
                  scoring_method,
                  pmi_lambda,
                  negative_permutation,
                  no_inference):
        config = ConfigManager(
            export_dir='{}/logit'.format(export_dir),
            test=test,
            model=self.model_name,
            max_length=self.lm.max_length,
            data=data,
            template_type=template_type,
            scoring_method=scoring_method,
            pmi_lambda=pmi_lambda
        )
        data_instance = AnalogyData(
            data=data, negative_permutation=negative_permutation, marginalize_permutation=scoring_method in ['ppl_pmi'])

        def prediction(positive: bool = True):
            prefix = 'positive' if positive else 'negative'
            cached_score = config.flatten_score[prefix]
            logging.info('{} permutation'.format(prefix))
            if cached_score:
                logging.info(' * load score')
                return cached_score

            input_data = data_instance.flatten_pos if positive else data_instance.flatten_neg

            assert not no_inference, '"no_inference==True" but no cache found'
            logging.info('# run scoring: {}'.format(scoring_method))
            shared = {'batch_size': batch_size, 'template_type': template_type}
            if scoring_method in ['ppl_pmi', 'ppl']:
                logging.info(' * ppl computation')
                full_score = self.lm.get_perplexity(word=input_data, **shared)
            elif scoring_method in ['ppl_tail_masked', 'ppl_head_masked']:
                logging.info(' * ppl computation ({})'.format(scoring_method))
                if scoring_method == 'ppl_tail_masked':
                    mask_index_condition = [-1] * len(input_data)
                else:
                    mask_index_condition = [-2] * len(input_data)
                full_score = self.lm.get_perplexity(
                    word=input_data, mask_index_condition=mask_index_condition, **shared)
            elif scoring_method == 'embedding_similarity':
                logging.info(' * embedding similarity')
                full_score = self.lm.get_embedding_similarity(word=input_data, **shared)
            elif scoring_method == 'pmi':
                score_list = []
                for n, (i, k) in enumerate(list(permutations(range(4), 2))):
                    logging.info(' * PMI permutation: {}, {} ({}/12)'.format(i, k, n + 1))
                    key = '{}-{}'.format(i, k)
                    if config.pmi_logits[prefix] and key in config.pmi_logits[prefix].keys():
                        _score = config.pmi_logits[prefix][key]
                        continue

                    mask_index = list(map(lambda x: x[i], input_data))
                    mask_index_condition = list(map(lambda x: x[k], input_data))
                    _score = self.lm.get_negative_pmi(
                        word=input_data, mask_index=mask_index, mask_index_condition=mask_index_condition,
                        weight=pmi_lambda, **shared)
                    config.cache_scores_pmi(key, _score, positive=positive)
                    score_list.append(_score)

                full_score = list(zip(*score_list))
            else:
                raise ValueError('unknown method: {}'.format(scoring_method))
            config.cache_scores(full_score, positive=positive)
            return full_score

        # run inference
        score_pos = prediction()
        config.cache_scores(score_pos)

        if negative_permutation:
            score_neg = prediction(positive=False)
            config.cache_scores(score_neg, positive=False)
        else:
            score_neg = None
        return score_pos, score_neg, config, data_instance
