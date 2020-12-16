import logging
from typing import List, Dict
from time import time
from itertools import permutations
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
                     scoring_method_config: Dict = None,
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
        :param scoring_method_config:
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
            scoring_method_config=scoring_method_config,
            model=self.model_name, max_length=self.lm.max_length, path_to_data=path_to_data,
            scoring_method=scoring_method, template_types=template_types, permutation_negative=permutation_negative,
            aggregation_positive=aggregation_positive, aggregation_negative=aggregation_negative
        )
        if config.output_exist and not overwrite_output:
            logging.info('skip as the output is already produced: {}'.format(config.export_dir))
            return

        # fetch data
        logging.info('fetch data and templating: {}'.format(path_to_data))
        # get data with all permutation regardless of the configuration
        data_instance = AnalogyData(path_to_data, template_types, permutation_negative=permutation_negative)

        # create batch
        logging.info('creating batch (data size: {})'.format(len(data_instance.flatten_prompt_pos)))

        def prediction(positive: bool = True):
            logging.info('{} permutation'.format('positive' if positive else 'negative'))
            prompt, relation = data_instance.get_prompt(positive=positive)
            cached_score = config.flatten_score_positive if positive else config.flatten_score_negative
            if prompt is None:
                logging.info(' * skip permutation')
                return None

            if cached_score:
                logging.info(' * load score')
                return cached_score

            assert not no_inference, '"no_inference==True" but no cache found'
            logging.info(' * run scoring: {}'.format(scoring_method))
            if scoring_method == 'ppl':
                return self.lm.get_perplexity(prompt, batch_size=batch_size)
            elif scoring_method == 'embedding_similarity':
                return self.lm.get_embedding_similarity(prompt, tokens_to_embed=relation, batch_size=batch_size)
            elif scoring_method == 'pmi':
                score_list = []
                for n, (i, k) in enumerate(list(permutations(range(4), 2))):
                    logging.info(' * PMI permutation: {}, {} ({}/12)'.format(i, k, n + 1))
                    tokens_to_mask = list(map(lambda x: x[i], relation))
                    tokens_to_condition = list(map(lambda x: x[k], relation))
                    _score = self.lm.get_negative_pmi(prompt,
                                                      batch_size=batch_size,
                                                      tokens_to_mask=tokens_to_mask,
                                                      tokens_to_condition=tokens_to_condition)
                    self.release_cache()
                    score_list.append(_score)
                return list(zip(*score_list))
            else:
                raise ValueError('unknown method: {}'.format(scoring_method))

        # run inference
        score_pos = prediction(positive=True)
        score_neg = prediction(positive=False)

        config.cache_scores(flatten_score_positive=score_pos, flatten_score_negative=score_neg)
        if skip_scoring_prediction:
            return

        # scoring method depending post aggregation
        if scoring_method == 'pmi':

            # we use same aggregation method for both positive/negative permutations
            assert len(scoring_method_config) == 1, 'unknown config: {}'.format(scoring_method_config)
            pmi_aggregation = scoring_method_config['aggregation']
            logging.info("PMI aggregator: {}".format(pmi_aggregation))
            aggregator = AGGREGATOR[pmi_aggregation]
            score_pos = list(map(lambda x: aggregator(x), score_pos))
            if score_neg:
                score_neg = list(map(lambda x: aggregator(x), score_neg))
        else:
            assert scoring_method_config is None, 'method {} has no configuration'.format(scoring_method)

        # restore the nested structure
        logging.info('restore batch structure')
        score = data_instance.insert_score(score_pos, score_neg)
        # score = restore_structure(list_nested_sentence, score_pos, batch_id_pos, score_neg, batch_id_neg)
        logit_pn = list(map(lambda o: list(map(lambda s: (aggregator_pos(s[0]), aggregator_neg(s[1])), o)), score))
        logit = list(map(lambda o: list(map(lambda s: s[1] - s[0], o)), logit_pn))
        pred = list(map(lambda x: x.index(max(x)), logit))
        # compute accuracy
        assert len(pred) == len(data_instance.answer)
        accuracy = sum(map(lambda x: int(x[0] == x[1]), zip(pred, data_instance.answer))) / len(data_instance.answer)
        logging.info('accuracy: {}'.format(accuracy))
        config.save(accuracy=accuracy, logit_pn=logit_pn, logit=logit, prediction=pred)
        logging.info('experiment completed: {} sec in total'.format(time()-start))
