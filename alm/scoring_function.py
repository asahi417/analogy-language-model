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
from .data import AnalogyData, get_dataset_raw
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


def get_report(export_prefix, export_dir: str = './experiments_results', test: bool = False):
    if test:
        export_prefix = export_prefix + '.test'
    else:
        export_prefix = export_prefix + '.valid'
    file = '{}/summary/{}'.format(export_dir, export_prefix)
    assert os.path.exists('{}.csv'.format(file)), 'csv not found: {}'.format(file)
    df = pd.read_csv('{}.csv'.format(file), index_col=0)
    df = df.drop_duplicates()
    logging.info('df has {} rows'.format(len(df)))
    return df


def export_report(export_prefix, export_dir: str = './experiments_results', test: bool = False):

    if test:
        export_prefix = export_prefix + '.test'
    else:
        export_prefix = export_prefix + '.valid'
    file = '{}/summary/jsonlines/{}'.format(export_dir, export_prefix)
    logging.info('compile jsonlins `{0}.jsonl` to csv file `{0}.csv`'.format(file))
    # save as a csv
    with open('{}.jsonl'.format(file), 'r') as f:
        json_line = list(filter(None, map(lambda x: json.loads(x) if len(x) > 0 else None, f.read().split('\n'))))
    logging.info('jsonline with {} lines'.format(len(json_line)))
    if os.path.exists('{}.csv'.format(file)):
        df = pd.read_csv('{}.csv'.format(file), index_col=0)
        df_tmp = pd.DataFrame(json_line)
        df = pd.concat([df, df_tmp])
    else:
        df = pd.DataFrame(json_line)
    df = df.drop_duplicates()
    df = df.sort_values(by='accuracy', ascending=False)
    file = '{}/summary/{}'.format(export_dir, export_prefix)
    logging.info('df has {} rows'.format(len(df)))
    logging.info('top result: \n {}'.format(df.head()))
    logging.info('exporting...')
    df.to_csv('{}.csv'.format(file))


class GridSearch:

    def __len__(self):
        return len(self.all_config)

    def __init__(self,
                 shared_config,
                 scoring_method, score, data_instance,
                 ppl_pmi_aggregation, ppl_pmi_lambda, ppl_pmi_alpha, positive_permutation_aggregation,
                 negative_permutation_aggregation, negative_permutation_weight,
                 ppl_pmi_marginal_version,
                 export_prediction: bool = False):
        """ Grid Search Aggregator: multiprocessing-oriented """
        # global variables
        self.ppl_pmi_marginal_version = ppl_pmi_marginal_version
        self.shared_config = shared_config
        self.score = score
        self.scoring_method = scoring_method
        self.data_instance = data_instance
        self.export_prediction = export_prediction
        # local parameters for grid search
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

            def compute_ppl_pmi(ppl_scores):
                opt_length = len(ppl_scores) ** 0.5
                assert opt_length.is_integer(), 'something wrong'
                opt_length = int(opt_length)

                if all(s == 0 for s in ppl_scores):
                    return [0] * opt_length

                # conditional negative log likelihood
                negative_log_likelihood_cond_h = list(map(
                    lambda x: log(ppl_scores[opt_length * x + x] / sum(
                        ppl_scores[opt_length * x: opt_length * (x +1)]
                    )),
                    range(opt_length)))
                negative_log_likelihood_cond_t = list(map(
                    lambda x: log(ppl_scores[opt_length * x + x] / sum(
                        [ppl_scores[opt_length * y + x] for y in range(opt_length)]
                    )),
                    range(opt_length)))

                # marginal negative log likelihood
                ppl_out_option = list(map(
                    lambda x: sum(map(lambda y: ppl_scores[x + opt_length * y], range(opt_length))),
                    range(opt_length)))
                negative_log_likelihood_mar_t = list(map(lambda x: log(x / sum(ppl_out_option)), ppl_out_option))

                ppl_out_option = list(map(
                    lambda x: sum(ppl_scores[x * opt_length: (x + 1) * opt_length]),
                    range(opt_length)))
                negative_log_likelihood_mar_h = list(map(lambda x: log(x / sum(ppl_out_option)), ppl_out_option))

                # negative pmi approx by perplexity difference: higher is better
                neg_pmi = list(map(
                    lambda x: aggregator([
                        x[0] * ppl_pmi_lambda - x[1] * ppl_pmi_alpha,
                        x[2] * ppl_pmi_lambda - x[3] * ppl_pmi_alpha
                    ]),
                    zip(negative_log_likelihood_cond_h, negative_log_likelihood_mar_h,
                        negative_log_likelihood_cond_t, negative_log_likelihood_mar_t)))
                return neg_pmi

            def compute_ppl_mar(ppl_scores):
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

            if not self.ppl_pmi_marginal_version:
                score = list(map(lambda o: (
                    list(map(lambda x: compute_ppl_pmi(list(map(lambda s: s[0][x], o))), range(8))),
                    list(map(lambda x: compute_ppl_pmi(list(map(lambda s: s[1][x] if len(s[1]) != 0 else 0, o))), range(16)))
                ), self.score))
            else:
                score = list(map(lambda o: (
                    list(map(lambda x: compute_ppl_mar(list(map(lambda s: s[0][x], o))), range(8))),
                    list(map(lambda x: compute_ppl_mar(list(map(lambda s: s[1][x] if len(s[1]) != 0 else 0, o))),
                             range(16)))
                ), self.score))

            logit_pn = list(map(
                lambda s: (
                    list(zip(
                        list(map(lambda o: aggregator_pos(o), list(zip(*s[0])))),
                        list(map(lambda o: aggregator_neg(o), list(zip(*s[1]))))
                    ))
                ),
                score))

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
        tmp_config = {
            'ppl_pmi_aggregation': ppl_pmi_aggregation,
            'ppl_pmi_lambda': ppl_pmi_lambda,
            'ppl_pmi_alpha': ppl_pmi_alpha,
            'positive_permutation_aggregation': positive_permutation_aggregation,
            'negative_permutation_aggregation': negative_permutation_aggregation,
            'negative_permutation_weight': negative_permutation_weight,
            'accuracy': accuracy}
        tmp_config.update(self.shared_config)
        if self.export_prediction:
            tmp_config['prediction'] = pred
        return tmp_config


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
        logging.debug('*** setting up a scorer ***')
        # language model setup
        self.max_length = max_length
        self.lm = TransformersLM(model=model, max_length=max_length, cache_dir=cache_dir, num_worker=num_worker)
        self.model_name = model

    def release_cache(self):
        if self.lm.device == "cuda":
            torch.cuda.empty_cache()

    def analogy_test(self,
                     data: str,
                     template_type: str,
                     export_prefix: str = 'main0',
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
                     ppl_pmi_alpha: (float, List) = 0.0,
                     skip_scoring_prediction: bool = False,
                     export_dir: str = './experiments_results',
                     no_inference: bool = False,
                     export_prediction: bool = False,
                     ppl_pmi_marginal_version: bool = False,
                     val_accuracy: float = None):
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
        shared_config = {
            'model': self.model_name,
            'max_length': self.max_length,
            'data': data,
            'template_type': template_type,
            'scoring_method': scoring_method,
            'negative_permutation': negative_permutation,
            'pmi_aggregation': pmi_aggregation,
            'pmi_lambda': pmi_lambda,
            'ppl_pmi_marginal_version': ppl_pmi_marginal_version
        }

        searcher = GridSearch(
            shared_config, scoring_method, score, data_instance,
            ppl_pmi_aggregation, ppl_pmi_lambda, ppl_pmi_alpha, positive_permutation_aggregation,
            negative_permutation_aggregation, negative_permutation_weight,
            ppl_pmi_marginal_version,
            export_prediction=export_prediction)
        logging.info('start grid search: {} combinations'.format(len(searcher)))
        logging.info('multiprocessing  : {} cpus'.format(os.cpu_count()))
        json_line = pool.map(searcher.single_run, searcher.index)
        pool.close()
        if val_accuracy is not None:
            for i in json_line:
                i['accuracy_validation'] = val_accuracy
        logging.info('export to {}/summary'.format(export_dir))
        os.makedirs('{}/summary'.format(export_dir), exist_ok=True)
        if test:
            export_prefix = export_prefix + '.test'
        else:
            export_prefix = export_prefix + '.valid'

        if export_prediction:
            logging.info('export prediction mode')
            assert len(json_line) == 1, 'more than one config found: {}'.format(len(searcher))
            # json_line = json.dumps(json_line[0])
            val_set, test_set = get_dataset_raw(data)
            data_raw = test_set if test else val_set
            prediction = json_line.pop('prediction')
            assert len(prediction) == len(data_raw), '{} != {}'.format(len(prediction), len(data_raw))
            for d, p in zip(data_raw, prediction):
                d['prediction'] = p
            os.makedirs('{}/summary/prediction_file'.format(export_dir), exist_ok=True)
            _file = '{}/summary/prediction_file/{}.prediction.{}.{}.{}.{}'.format(
                export_dir, export_prefix, data, self.model_name, scoring_method, ppl_pmi_marginal_version)
            pd.DataFrame(data_raw).to_csv('{}.csv'.format(_file))
            with open('{}.json'.format(_file), 'w') as f:
                json.dump(json_line, f)
            logging.info("prediction exported: {}".format(_file))
        else:
            # save as a json line
            os.makedirs('{}/summary/jsonlines'.format(export_dir), exist_ok=True)
            if os.path.exists('{}/summary/jsonlines/{}.jsonl'.format(export_dir, export_prefix)):
                with open('{}/summary/jsonlines/{}.jsonl'.format(export_dir, export_prefix), 'a') as writer:
                    writer.write('\n')
                    writer.write('\n'.join(list(map(lambda x: json.dumps(x), json_line))))
            else:
                with open('{}/summary/jsonlines/{}.jsonl'.format(export_dir, export_prefix), 'w') as writer:
                    writer.write('\n'.join(list(map(lambda x: json.dumps(x), json_line))))
        print(json_line)
        return list(map(lambda x: x['accuracy'], json_line))

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
            test=test, data=data, negative_permutation=negative_permutation,
            marginalize_permutation=scoring_method in ['ppl_pmi'])

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
            full_full_score = []
            if data == 'bats':
                s = int(len(input_data)/2)
                input_data = [input_data[:s], input_data[s:]]
            else:
                input_data = [input_data]
            for input_data_sub in input_data:
                if scoring_method in ['ppl_pmi', 'ppl']:
                    logging.info(' * ppl computation')
                    full_score = self.lm.get_perplexity(word=input_data_sub, **shared)
                elif scoring_method in ['ppl_tail_masked', 'ppl_head_masked', 'ppl_add_masked']:
                    logging.info(' * ppl computation ({})'.format(scoring_method))
                    if scoring_method == 'ppl_tail_masked':
                        full_score = self.lm.get_perplexity(
                            word=input_data_sub, mask_index_condition=[-1] * len(input_data_sub), **shared)
                    elif scoring_method == 'ppl_head_masked':
                        full_score = self.lm.get_perplexity(
                            word=input_data_sub, mask_index_condition=[-2] * len(input_data_sub), **shared)
                    else:
                        full_score_tail = self.lm.get_perplexity(
                            word=input_data_sub, mask_index_condition=[-1] * len(input_data_sub), **shared)
                        full_score_head = self.lm.get_perplexity(
                            word=input_data_sub, mask_index_condition=[-2] * len(input_data_sub), **shared)
                        full_score = list(map(lambda x: sum(x), zip(full_score_head, full_score_tail)))
                elif scoring_method == 'embedding_similarity':
                    logging.info(' * embedding similarity')
                    full_score = self.lm.get_embedding_similarity(word=input_data_sub, **shared)
                elif scoring_method == 'pmi':
                    score_list = []
                    for n, (i, k) in enumerate(list(permutations(range(4), 2))):
                        logging.info(' * PMI permutation: {}, {} ({}/12)'.format(i, k, n + 1))
                        key = '{}-{}'.format(i, k)
                        if config.pmi_logits[prefix] and key in config.pmi_logits[prefix].keys():
                            _score = config.pmi_logits[prefix][key]
                            continue
                        _score = self.lm.get_negative_pmi(
                            word=input_data_sub,
                            mask_index=[i] * len(input_data_sub),
                            mask_index_condition=[k] * len(input_data_sub),
                            weight=pmi_lambda, **shared)
                        config.cache_scores_pmi(key, _score, positive=positive)
                        score_list.append(_score)

                    full_score = list(zip(*score_list))
                else:
                    raise ValueError('unknown method: {}'.format(scoring_method))
                full_full_score += full_score
            config.cache_scores(full_full_score, positive=positive)
            return full_full_score

        # run inference
        score_pos = prediction()
        config.cache_scores(score_pos)

        if negative_permutation:
            score_neg = prediction(positive=False)
            config.cache_scores(score_neg, positive=False)
        else:
            score_neg = None
        return score_pos, score_neg, config, data_instance
