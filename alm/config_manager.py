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
        self.flatten_score = None
        if not os.path.exists(export_dir):
            self.export_dir = os.path.join(export_dir, get_random_string())
        else:
            ex_configs = {i: safe_open(i) for i in glob('{}/*/config.json'.format(export_dir))}
            # check duplication
            same_config = list(filter(lambda x: x[1] == self.config, ex_configs.items()))
            if len(same_config) != 0:
                raise ValueError('found same configuration in the directory: {}'.format(same_config[0][0]))

            # load model prediction if the model config is at least same, enabling to skip model inference in case
            cond = ['model', 'max_length', 'path_to_data', 'permutation_positive', 'permutation_negative',
                    'template_types', 'scoring_method']
            cond_config = list(filter(
                lambda x: ([x[1][i] for i in cond] == [self.config[i] for i in cond]) and os.path.exists(
                    x[0].replace('config.json', 'flatten_score.pkl')),
                ex_configs.items()))
            if len(cond_config) != 0:
                _file = cond_config[0][0].replace('config.json', 'flatten_score.pkl')
                with open(_file, "rb") as fp:  # Unpickling
                    self.flatten_score = pickle.load(fp)
                print(self.flatten_score)
                logging.info('load flatten_score from {}'.format(_file))

            # create new experiment directory
            ex = list(map(lambda x: x.replace('/config.json', '').split('/')[-1], ex_configs.keys()))
            self.export_dir = os.path.join(export_dir, get_random_string(exclude=ex))

    def save(self, accuracy: float, flatten_score: List, logit_pn: List, logit: List, prediction: List):
        """ export data """
        os.makedirs(self.export_dir, exist_ok=True)
        if self.flatten_score is None:
            with open('{}/flatten_score.pkl'.format(self.export_dir), "wb") as fp:
                pickle.dump(flatten_score, fp)
        with open('{}/accuracy.json'.format(self.export_dir), 'w') as f:
            json.dump({"accuracy": accuracy}, f)
        with open('{}/config.json'.format(self.export_dir), 'w') as f:
            json.dump(self.config, f)
        with open('{}/output.json'.format(self.export_dir), 'w') as f:
            json.dump({"logit": logit, "logit_pn": logit_pn, "prediction": prediction}, f)
        logging.info('saved at {}'.format(self.export_dir))




