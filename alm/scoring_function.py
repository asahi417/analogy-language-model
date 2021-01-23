import logging
import tqdm
import os
import json
from typing import List
from multiprocessing import Pool
from itertools import permutations, product
from math import log

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

import torch
import pandas as pd
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
PBAR = tqdm.tqdm()


def export_report(export_prefix, export_dir: str = './experiments_results'):
    # save as a csv
    with open('{}/summary/{}.jsonl'.format(export_dir, export_prefix), 'r') as f:
        json_line = list(filter(None, map(lambda x: json.loads(x) if len(x) > 0 else None, f.read().split('\n'))))

    if os.path.exists('{}/summary/{}.csv'.format(export_dir, export_prefix)):
        df = pd.read_csv('{}/summary/{}.csv'.format(export_dir, export_prefix), index_col=0)
        df_tmp = pd.DataFrame(json_line)
        df = pd.concat([df, df_tmp])
        df = df.drop_duplicates()
        df.to_csv('{}/summary/{}.csv'.format(export_dir, export_prefix))
    else:
        pd.DataFrame(json_line).to_csv('{}/summary/{}.csv'.format(export_dir, export_prefix))


class GridSearch:

    def __len__(self):
        return len(self.all_config)

    def __init__(self, scoring_method, score, data_instance,
                 ppl_pmi_aggregation, ppl_pmi_lambda, ppl_pmi_alpha, positive_permutation_aggregation,
                 negative_permutation_aggregation, negative_permutation_weight):
        """ Grid Search Aggregator: multiprocessing-oriented """
        # global variables
        self.score = score
        self.scoring_method = scoring_method
        self.data_instance = data_instance
        # local parameters for grid serach
        if type(ppl_pmi_aggregation) is not list:
            ppl_pmi_aggregation = [ppl_pmi_aggregation]
        if type(ppl_pmi_lambda) is not list:
            ppl_pmi_lambda = [ppl_pmi_lambda]
        if type(ppl_pmi_alpha) is not list:
            ppl_pmi_alpha = [ppl_pmi_alpha]
        if type(positive_permutation_aggregation) is not list:
            positive_permutation_aggregation = [positive_permutation_aggregation]
        if type(negative_permutation_aggregation) is not list:
            negative_permutation_aggregation = [negative_permutation_aggregation]
        if type(negative_permutation_weight) is not list:
            negative_permutation_weight = [negative_permutation_weight]
        self.all_config = list(product(
            ppl_pmi_aggregation, ppl_pmi_lambda, ppl_pmi_alpha, positive_permutation_aggregation,
            negative_permutation_aggregation, negative_permutation_weight
        ))
        self.index = list(range(len(self.all_config)))

    def single_run(self, config_index: int):
        PBAR.update(1)
        ppl_pmi_aggregation, ppl_pmi_lambda, ppl_pmi_alpha, positive_permutation_aggregation,\
            negative_permutation_aggregation, negative_permutation_weight = self.all_config[config_index]

        aggregator_pos = AGGREGATOR[positive_permutation_aggregation]
        aggregator_neg = AGGREGATOR[negative_permutation_aggregation]

        # ppl_pmi aggregation
        if self.scoring_method == 'ppl_pmi':
            assert ppl_pmi_aggregation is not None
            aggregator = AGGREGATOR[ppl_pmi_aggregation]

            def compute_pmi(ppl_scores):
                opt_length = len(ppl_scores) ** 0.5
                assert opt_length.is_integer(), 'something wrong'
                opt_length = int(opt_length)

                if all(s == 0 for s in ppl_scores):
                    return [0] * opt_length

                # conditional negative log likelihood (fixed head and tail tokens)
                ppl_in_option = list(map(lambda x: ppl_scores[opt_length * x + x], range(opt_length)))
                negative_log_likelihood_cond = list(map(lambda x: log(x / sum(ppl_in_option)), ppl_in_option))

                # marginal negative log likelihood (tail token)
                ppl_out_option = list(map(
                    lambda x: sum(map(lambda y: ppl_scores[x + opt_length * y], range(opt_length))),
                    range(opt_length)))
                negative_log_likelihood_mar_t = list(map(lambda x: log(x / sum(ppl_out_option)), ppl_out_option))

                # marginal negative log likelihood (head token)
                ppl_out_option = list(map(
                    lambda x: sum(ppl_scores[x * opt_length: (x + 1) * opt_length]),
                    range(opt_length)))
                negative_log_likelihood_mar_h = list(map(lambda x: log(x / sum(ppl_out_option)), ppl_out_option))

                # negative pmi approx by perplexity difference: higher is better
                neg_pmi = list(map(
                    lambda x: x[0] * ppl_pmi_lambda - aggregator([x[1], x[2]]) * ppl_pmi_alpha,
                    zip(negative_log_likelihood_cond, negative_log_likelihood_mar_h, negative_log_likelihood_mar_t)))
                return neg_pmi

            pmi = list(map(lambda o: (
                list(map(lambda x: compute_pmi(list(map(lambda s: s[0][x], o))), range(8))),
                list(map(lambda x: compute_pmi(list(map(lambda s: s[1][x] if len(s[1]) != 0 else 0, o))), range(16)))
            ), self.score))

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
                self.score))

        logit = list(map(lambda o: list(map(lambda s: negative_permutation_weight * s[1] - s[0], o)), logit_pn))
        pred = list(map(lambda x: x.index(max(x)), logit))

        # compute accuracy
        label = self.data_instance.answer
        assert len(pred) == len(label)
        accuracy = sum(map(lambda x: int(x[0] == x[1]), zip(pred, label))) / len(label)
        # config.save(accuracy=accuracy, logit_pn=logit_pn, logit=logit, prediction=pred)
        return {'ppl_pmi_aggregation': ppl_pmi_aggregation,
                'ppl_pmi_lambda': ppl_pmi_lambda,
                'ppl_pmi_alpha': ppl_pmi_alpha,
                'positive_permutation_aggregation': positive_permutation_aggregation,
                'negative_permutation_aggregation': negative_permutation_aggregation,
                'negative_permutation_weight': negative_permutation_weight,
                'accuracy': accuracy}


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
                     export_prefix: str = 'main',
                     no_inference: bool = False):
        """ relation scoring test on analogy dataset

        :param data:
        :param test:
        :param scoring_method:
        :param ppl_pmi_aggregation: upto p_1
        :param pmi_aggregation: upto p_11
        :param positive_permutation_aggregation: upto p_7
        :param negative_permutation_aggregation: upto p_15
        :param pmi_lambda:
        :param ppl_pmi_lambda:
        :param batch_size:
        :param template_type: a list of templates for prompting
        :param skip_scoring_prediction:
        :param no_inference: use only cached score
        :param export_dir: directory to export the result
        :return:
        """
        ##############
        # get scores #
        ##############
        logging.info('get LM score')
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

        ##################
        # restore scores #
        ##################
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
            assert pmi_lambda == 1.0 and pmi_aggregation is None, 'pmi_lambda/pmi_aggregation should be default'
        # restore the nested structure
        logging.info('re-format LM score')
        score = data_instance.insert_score(score_pos, score_neg)

        ###############
        # grid search #
        ###############
        pool = Pool()
        searcher = GridSearch(
            scoring_method, score, data_instance,
            ppl_pmi_aggregation, ppl_pmi_lambda, ppl_pmi_alpha, positive_permutation_aggregation,
            negative_permutation_aggregation, negative_permutation_weight)
        logging.info('start grid search: {} combinations'.format(len(searcher)))
        logging.info('multiprocessing  : {} cpus'.format(os.cpu_count()))
        json_line = pool.map(searcher.single_run, searcher.index)
        pool.close()
        logging.info('export to {}/summary'.format(export_dir))
        os.makedirs('{}/summary'.format(export_dir), exist_ok=True)
        export_prefix = export_prefix + '.test' if test else '.valid'
        # save as a json line
        if os.path.exists('{}/summary/{}.jsonl'.format(export_dir, export_prefix)):
            with open('{}/summary/{}.jsonl'.format(export_dir, export_prefix), 'a') as writer:
                writer.write('\n'.join(list(map(lambda x: json.dumps(x), json_line))))
        else:
            with open('{}/summary/{}.jsonl'.format(export_dir, export_prefix), 'w') as writer:
                writer.write('\n'.join(list(map(lambda x: json.dumps(x), json_line))))

    def get_score(self, export_dir, test, template_type, data, batch_size, scoring_method, pmi_lambda,
                  negative_permutation, no_inference):
        config = ConfigManager(
            export_dir=export_dir,
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
            elif scoring_method in ['ppl_tail_masked', 'ppl_head_masked', 'ppl_add_masked']:
                logging.info(' * ppl computation ({})'.format(scoring_method))
                if scoring_method == 'ppl_tail_masked':
                    full_score = self.lm.get_perplexity(
                        word=input_data, mask_index_condition=[-1] * len(input_data), **shared)
                elif scoring_method == 'ppl_head_masked':
                    full_score = self.lm.get_perplexity(
                        word=input_data, mask_index_condition=[-2] * len(input_data), **shared)
                else:
                    full_score_head = self.lm.get_perplexity(
                        word=input_data, mask_index_condition=[-1] * len(input_data), **shared)
                    full_score_tail = self.lm.get_perplexity(
                        word=input_data, mask_index_condition=[-2] * len(input_data), **shared)
                    full_score = full_score_head + full_score_tail
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
