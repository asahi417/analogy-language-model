import logging
from typing import List
from time import time
from itertools import permutations
from math import log
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

import torch
from . import TransformersLM
from .data import AnalogyData
from .config_manager import ConfigManager

AGGREGATOR = {
    'mean': lambda x: sum(x)/len(x), 'max': lambda x: max(x), 'min': lambda x: min(x),
    'p_0': lambda x: x[0], 'p_1': lambda x: x[1], 'p_2': lambda x: x[2], 'p_3': lambda x: x[3],
    'p_4': lambda x: x[4], 'p_5': lambda x: x[5], 'p_6': lambda x: x[6], 'p_7': lambda x: x[7],
    'p_8': lambda x: x[8], 'p_9': lambda x: x[9], 'p_10': lambda x: x[10], 'p_11': lambda x: x[11],
    'none': lambda x: 0
}


class RelationScorer:
    """ Scoring relations with language models """

    def __init__(self,
                 model: str = 'roberta-base',
                 max_length: int = 32,  # the template is usually a short sentence
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
        self.lm = TransformersLM(
            model=model,
            max_length=max_length,
            cache_dir=cache_dir,
            num_worker=num_worker)
        self.model_name = model

    def release_cache(self):
        if self.lm.device == "cuda":
            torch.cuda.empty_cache()

    def analogy_test(self,
                     path_to_data: str,
                     batch_size: int = 4,
                     scoring_method: str = 'ppl',
                     pmi_aggregation: str = None,
                     pmi_lambda: float = 1.0,
                     ppl_pmi_lambda: float = 1.0,
                     template_types: List = None,
                     permutation_negative: bool = False,
                     aggregation_positive: str = 'mean',
                     aggregation_negative: str = 'none',
                     skip_scoring_prediction: bool = False,
                     export_dir: str = './results',
                     no_inference: bool = False,
                     overwrite_output: bool = False):
        """ relation scoring test on analogy dataset

        :param path_to_data:
        :param scoring_method:
        :param pmi_aggregation:
        :param pmi_lambda:
        :param ppl_pmi_lambda:
        :param batch_size:
        :param template_types: a list of templates for prompting
        :param permutation_negative: if utilize negative permutation
        :param aggregation_positive: aggregation method for positive permutations (`mean`, `max`, `min`)
        :param aggregation_negative: aggregation method for negative permutations (`mean`, `max`, `min`)
        :param skip_scoring_prediction:
        :param no_inference: use only cached score
        :param export_dir: directory to export the result
        :param overwrite_output: overwrite existing result file
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
            pmi_aggregation=pmi_aggregation,
            pmi_lambda=pmi_lambda,
            model=self.model_name, max_length=self.lm.max_length, path_to_data=path_to_data,
            scoring_method=scoring_method, template_types=template_types, permutation_negative=permutation_negative,
            aggregation_positive=aggregation_positive, aggregation_negative=aggregation_negative,
            ppl_pmi_lambda=ppl_pmi_lambda
        )
        if config.output_exist and not overwrite_output:
            logging.info('skip as the output is already produced: {}'.format(config.export_dir))
            return

        # fetch data
        logging.info('fetch data and templating: {}'.format(path_to_data))
        # get data with all permutation regardless of the configuration
        permutation_marginalize = scoring_method in ['ppl_pmi']
        data_instance = AnalogyData(path_to_data, template_types,
                                    permutation_negative=permutation_negative,
                                    permutation_marginalize=permutation_marginalize)

        # create batch
        logging.info('creating batch (data size: {})'.format(len(data_instance.flatten_prompt_pos)))

        def prediction(positive: bool = True):
            prefix = 'positive' if positive else 'negative'
            logging.info('{} permutation'.format(prefix))
            prompt, relation = data_instance.get_prompt(positive=positive)
            cached_score = config.flatten_score[prefix]
            assert prompt

            if cached_score:
                logging.info(' * load score')
                return cached_score

            assert not no_inference, '"no_inference==True" but no cache found'
            logging.info('# run scoring: {}'.format(scoring_method))
            if scoring_method in ['ppl_pmi', 'pmi']:
                logging.info(' * ppl computation')
                full_score = self.lm.get_perplexity(prompt, batch_size=batch_size)
            elif scoring_method == 'embedding_similarity':
                full_score = self.lm.get_embedding_similarity(prompt, tokens_to_embed=relation, batch_size=batch_size)
            elif scoring_method == 'pmi':
                score_list = []
                for n, (i, k) in enumerate(list(permutations(range(4), 2))):
                    logging.info(' * PMI permutation: {}, {} ({}/12)'.format(i, k, n + 1))
                    key = '{}-{}'.format(i, k)
                    if config.pmi_logits[prefix] and key in config.pmi_logits[prefix].keys():
                        _score = config.pmi_logits[prefix][key]
                        continue

                    tokens_to_mask = list(map(lambda x: x[i], relation))
                    tokens_to_condition = list(map(lambda x: x[k], relation))
                    _score = self.lm.get_negative_pmi(prompt,
                                                      batch_size=batch_size,
                                                      tokens_to_mask=tokens_to_mask,
                                                      tokens_to_condition=tokens_to_condition,
                                                      weight=pmi_lambda)
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

        if permutation_negative:
            score_neg = prediction(positive=False)
            config.cache_scores(score_neg, positive=False)
        else:
            score_neg = None

        if skip_scoring_prediction:
            return

        # negative pmi score aggregation
        if scoring_method == 'pmi':
            assert pmi_aggregation, 'undefined pmi aggregation'
            # we use same aggregation method for both positive/negative permutations
            logging.info("PMI aggregator: {}".format(pmi_aggregation))
            aggregator = AGGREGATOR[pmi_aggregation]
            score_pos = list(map(lambda x: aggregator(x), score_pos))
            if score_neg:
                score_neg = list(map(lambda x: aggregator(x), score_neg))
        else:
            assert pmi_lambda == 1.0 and pmi_aggregation, 'pmi_lambda/pmi_aggregation should be default'

        # restore the nested structure
        logging.info('restore batch structure')
        score = data_instance.insert_score(score_pos, score_neg)

        # ppl_pmi aggregation
        if scoring_method == 'ppl_pmi':

            def compute_pmi(ppl_scores):
                opt_length = len(ppl_scores) ** 0.5
                assert opt_length.is_integer(), 'something wrong'
                opt_length = int(opt_length)

                if all(s == 0 for s in ppl_scores):
                    return [0] * opt_length

                # conditional negative log likelihood (fixed head and tail tokens)
                ppl_in_option = list(map(lambda x: ppl_scores[opt_length * x + x], range(opt_length)))
                negative_log_likelihood_cond = list(map(lambda x: log(x / sum(ppl_in_option)), ppl_in_option))

                # marginal negative log likelihood (fixed tail token)
                ppl_out_option = list(map(
                    lambda x: sum(map(lambda y: ppl_scores[x + opt_length * y], range(opt_length))),
                    range(opt_length)))
                negative_log_likelihood_mar = list(map(lambda x: log(x / sum(ppl_out_option)), ppl_out_option))

                # negative pmi approx by perplexity difference: higher is better
                neg_pmi = list(map(
                    lambda x: x[0] - x[1] * ppl_pmi_lambda, zip(negative_log_likelihood_cond, negative_log_likelihood_mar)))
                return neg_pmi

            # loop over all positive permutations
            pmi = list(map(lambda o: (
                list(map(lambda x: compute_pmi(list(map(lambda s: s[0][x], o))), range(8))),
                list(map(lambda x: compute_pmi(list(map(lambda s: s[1][x] if len(s[1]) != 0 else 0, o))), range(8)))
            ), score))

            logit_pn = list(map(
                lambda s: (
                    list(zip(
                        list(map(lambda o: aggregator_pos(o), list(zip(*s[0])))),
                        list(map(lambda o: aggregator_neg(o), list(zip(*s[1]))))
                    ))
                ),
                pmi))

        else:
            logit_pn = list(map(
                lambda o: list(map(
                    lambda s: (aggregator_pos(s[0]), aggregator_neg(s[1])),
                    o)),
                score))
        logit = list(map(lambda o: list(map(lambda s: s[1] - s[0], o)), logit_pn))
        pred = list(map(lambda x: x.index(max(x)), logit))

        # compute accuracy
        assert len(pred) == len(data_instance.answer)
        accuracy = sum(map(lambda x: int(x[0] == x[1]), zip(pred, data_instance.answer))) / len(data_instance.answer)
        logging.info('accuracy: {}'.format(accuracy))
        config.save(accuracy=accuracy, logit_pn=logit_pn, logit=logit, prediction=pred)
        logging.info('experiment completed: {} sec in total'.format(time()-start))
