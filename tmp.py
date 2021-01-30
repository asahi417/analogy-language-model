import json
import pickle

from glob import glob
from itertools import product

all_templates = ['is-to-what', 'is-to-as', 'rel-same', 'what-is-to', 'she-to-as', 'as-what-same']
data = ['sat', 'u2', 'u4', 'google', 'bats']
model = ['roberta-large', 'bert-cased-large']


def get_config(path):
    with open('{}/config.json'.format(path), 'r') as f:
        config = json.load(f)
        return config['template_type']


for d, m in product(data, model):
    a = {get_config(i): i for i in glob('./experiments_results/logit/{}/{}/ppl_add_masked/*'.format(d, m))}
    h = {get_config(i): i for i in glob('./experiments_results/logit/{}/{}/ppl_head_masked/*'.format(d, m))}
    t = {get_config(i): i for i in glob('./experiments_results/logit/{}/{}/ppl_tail_masked/*'.format(d, m))}
    print(a)
    print(h)
    print(t)
    for tmp in all_templates:
        with open('{}/flatten_score.positive.valid.pkl'.format(t[tmp]), 'rb') as fp:
            t_v = pickle.load(fp)
        with open('{}/flatten_score.positive.valid.pkl'.format(h[tmp]), 'rb') as fp:
            h_v = pickle.load(fp)
        a_v = list(map(lambda x: sum(x), zip(t_v, h_v)))
        with open('{}/flatten_score.positive.valid.pkl'.format(a[tmp]), 'wb') as fp:
            pickle.dump(a_v, fp)

