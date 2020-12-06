""" configuration manager for `scoring_function.RelationScorer` """
import os
import random
import json
import string
import logging
import pickle
from glob import glob
from typing import List
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def get_random_string(length: int = 6, exclude: List = None):
    tmp = ''.join(random.choice(string.ascii_lowercase) for _ in range(length))
    if exclude:
        while tmp in exclude:
            tmp = ''.join(random.choice(string.ascii_lowercase) for _ in range(length))
    return tmp


def safe_open(_file):
    with open(_file, 'r') as f:
        return json.load(f)


class ConfigManager:
    """ configuration manager for `scoring_function.RelationScorer` """

    def __init__(self, export_dir: str, **kwargs):
        """ configuration manager for `scoring_function.RelationScorer` """
        self.config = kwargs
        logging.info('*** setting up a config manager ***\n' +
                     '\n'.join(list(map(lambda x: '{} : {}'.format(x[0], x[1]), self.config.items()))))
        cache_dir = os.path.join(export_dir, 'flatten_scores')
        export_dir = os.path.join(export_dir, 'outputs')
        self.flatten_score_positive = None
        self.flatten_score_negative = None
        if not os.path.exists(export_dir):
            self.export_dir = os.path.join(export_dir, get_random_string())
        else:
            ex_configs = {i: safe_open(i) for i in glob('{}/*/config.json'.format(export_dir))}
            # check duplication
            same_config = list(filter(lambda x: x[1] == self.config, ex_configs.items()))
            if len(same_config) != 0:
                raise ValueError('found same configuration in the directory: {}'.format(same_config[0][0]))
            # create new experiment directory
            ex = list(map(lambda x: x.replace('/config.json', '').split('/')[-1], ex_configs.keys()))
            self.export_dir = os.path.join(export_dir, get_random_string(exclude=ex))

        # load model prediction if the model config is at least same, enabling to skip model inference in case
        cond = ['model', 'max_length', 'path_to_data', 'template_types', 'scoring_method']
        self.config_cache = {k: v for k, v in self.config.items() if k in cond}
        ex_configs = {i: safe_open(i) for i in glob('{}/*/config.json'.format(cache_dir))}
        same_config = list(filter(lambda x: x[1] == self.config_cache, ex_configs.items()))
        if len(same_config) != 0:
            self.cache_dir = same_config[0][0].replace('config.json', '')

            _file = os.path.join(self.cache_dir, 'flatten_score_positive.pkl')
            with open(_file, "rb") as fp:  # Unpickling
                self.flatten_score_positive = pickle.load(fp)
            logging.info('load flatten_score_positive from {}'.format(_file))

            _file = os.path.join(self.cache_dir, 'flatten_score_negative.pkl')
            if os.path.exists(_file):
                with open(_file, "rb") as fp:  # Unpickling
                    self.flatten_score_negative = pickle.load(fp)
                logging.info('load flatten_score_negative from {}'.format(_file))
        else:
            self.cache_dir = os.path.join(cache_dir, get_random_string())

    def cache_scores(self, flatten_score_positive: List, flatten_score_negative: List = None):
        """ cache scores """
        os.makedirs(self.cache_dir, exist_ok=True)
        if not os.path.exists('{}/config.json'.format(self.cache_dir)):
            with open('{}/config.json'.format(self.cache_dir), 'w') as f:
                json.dump(self.config_cache, f)
        if self.flatten_score_positive is None:
            with open('{}/flatten_score_positive.pkl'.format(self.cache_dir), "wb") as fp:
                pickle.dump(flatten_score_positive, fp)
        if self.flatten_score_negative is None and flatten_score_negative is not None:
            with open('{}/flatten_score_negative.pkl'.format(self.cache_dir), "wb") as fp:
                pickle.dump(flatten_score_negative, fp)

    def save(self, accuracy: float, logit_pn: List, logit: List, prediction: List):
        """ export data """
        os.makedirs(self.export_dir, exist_ok=True)
        with open('{}/accuracy.json'.format(self.export_dir), 'w') as f:
            json.dump({"accuracy": accuracy}, f)
        with open('{}/config.json'.format(self.export_dir), 'w') as f:
            json.dump(self.config, f)
        with open('{}/output.json'.format(self.export_dir), 'w') as f:
            json.dump({"logit": logit, "logit_pn": logit_pn, "prediction": prediction}, f)
        logging.info('saved at {}'.format(self.export_dir))




