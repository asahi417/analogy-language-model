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

    def __init__(self, export_dir: str, test: bool, **kwargs):
        """ configuration manager for `scoring_function.RelationScorer` """
        self.config = kwargs
        self.prefix = 'test' if test else 'valid'
        logging.info('*** setting up a config manager ***\n' +
                     '\n'.join(list(map(lambda x: '{} : {}'.format(x[0], x[1]), self.config.items()))))
        cache_dir = os.path.join(export_dir, kwargs['data'], kwargs['model'], kwargs['scoring_method'])
        # if not os.path.exists(export_dir):
        #     self.export_dir = os.path.join(export_dir, kwargs['data'], get_random_string())
        # else:
        #     # this is going to be very large loop
        #     ex_configs = {i: safe_open(i) for i in glob('{}/*/config.json'.format(export_dir))}
        #     # check duplication
        #     same_config = list(filter(lambda x: x[1] == self.config, ex_configs.items()))
        #     if len(same_config) != 0:
        #         logging.debug("found same configuration: {}".format(same_config[0][0]))
        #         self.output_exist = True
        #         self.export_dir = same_config[0][0].replace('/config.json', '')
        #     else:
        #         # create new experiment directory
        #         ex = list(map(lambda x: x.replace('/config.json', '').split('/')[-1], ex_configs.keys()))
        #         self.export_dir = os.path.join(export_dir, get_random_string(exclude=ex))

        # if skip_flatten_score:
        #     self.flatten_score = None
        #     self.flatten_score_mar = None
        #     self.cache_dir = None
        # else:
        self.pmi_logits = {'positive': None, 'negative': None}
        self.flatten_score = {'positive': None, 'negative': None}
        self.flatten_score_mar = {'positive': None, 'negative': None}

        # load model prediction if the model config is at least same, enabling to skip model inference in case
        # if not os.path.exists(cache_dir):
        #     os.makedirs(cache_dir, exist_ok=True)
        # else:
        ex_configs = {i: safe_open(i) for i in glob('{}/*/config.json'.format(cache_dir))}
        same_config = list(filter(lambda x: x[1] == self.config, ex_configs.items()))
        if len(same_config) != 0:
            assert len(same_config) == 1, 'duplicated config found {}'.format(same_config)
            self.cache_dir = same_config[0][0].replace('config.json', '')

            # load intermediate score
            for i in ['positive', 'negative']:
                _file = os.path.join(self.cache_dir, 'flatten_score.{}.{}.pkl'.format(i, self.prefix))
                if os.path.exists(_file):
                    with open(_file, "rb") as fp:  # Unpickling
                        self.flatten_score[i] = pickle.load(fp)
                    logging.debug('load flatten_score.{} from {}'.format(i, _file))

                # load stats for ppl_pmi
                _file = os.path.join(self.cache_dir, 'flatten_score_mar.{}.{}.pkl'.format(i, self.prefix))
                if os.path.exists(_file):
                    with open(_file, "rb") as fp:  # Unpickling
                        self.flatten_score_mar[i] = pickle.load(fp)
                    logging.debug('load flatten_score_mar.{} from {}'.format(i, _file))

            # load intermediate score for PMI specific
            if self.config['scoring_method'] in ['pmi']:
                for i in ['positive', 'negative']:
                    # skip if full score is loaded
                    if i in self.flatten_score.keys():
                        continue
                    self.pmi_logits[i] = {}
                    for _file in glob(os.path.join(self.cache_dir, 'pmi.{}.*.{}pkl'.format(i, self.prefix))):
                        if os.path.exists(_file):
                            k = _file.split('pmi.{}.'.format(i))[-1].replace('.pkl', '')
                            with open(_file, "rb") as fp:  # Unpickling
                                self.pmi_logits[i][k] = pickle.load(fp)
                            logging.debug('load pmi.{} from {}'.format(i, _file))

        else:
            self.cache_dir = os.path.join(cache_dir, get_random_string())

    def __cache_init(self):
        assert self.cache_dir is not None
        os.makedirs(self.cache_dir, exist_ok=True)
        if not os.path.exists('{}/config.json'.format(self.cache_dir)):
            with open('{}/config.json'.format(self.cache_dir), 'w') as f:
                json.dump(self.config, f)

    def cache_scores_pmi(self, logit_name: str, pmi_logit: List, positive: bool = True):
        self.__cache_init()
        prefix = 'positive' if positive else 'negative'
        with open('{}/pmi.{}.{}.{}.pkl'.format(self.cache_dir, prefix, logit_name, self.prefix), "wb") as fp:
            pickle.dump(pmi_logit, fp)

    def cache_scores(self, flatten_score: List, positive: bool = True):
        """ cache scores """
        self.__cache_init()
        prefix = 'positive' if positive else 'negative'
        with open('{}/flatten_score.{}.{}.pkl'.format(self.cache_dir, prefix, self.prefix), "wb") as fp:
            pickle.dump(flatten_score, fp)

    # def save(self, accuracy: float, logit_pn: List, logit: List, prediction: List):
    #     """ export data """
    #     if os.path.exists(self.export_dir):
    #         shutil.rmtree(self.export_dir)
    #     os.makedirs(self.export_dir, exist_ok=True)
    #     with open('{}/accuracy.json'.format(self.export_dir), 'w') as f:
    #         json.dump({"accuracy": accuracy}, f)
    #     with open('{}/config.json'.format(self.export_dir), 'w') as f:
    #         json.dump(self.config, f)
    #     with open('{}/output.json'.format(self.export_dir), 'w') as f:
    #         json.dump({"logit": logit, "logit_pn": logit_pn, "prediction": prediction}, f)
    #     logging.debug('saved at {}'.format(self.export_dir))




