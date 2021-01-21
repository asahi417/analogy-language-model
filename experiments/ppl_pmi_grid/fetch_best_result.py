import json
import os
from glob import glob
from pprint import pprint

import pandas as pd


def safe_open(_file, keys):
    with open(_file, 'r') as f:
        d = json.load(f)
        return {k: d[k_] for k_ in keys}


def get_best_config(data):
    df = pd.read_csv('./experiments/ppl_pmi_grid/results/summary.{}.csv'.format(data), index_col=0)
    df = df.sort_values(by='accuracy', ascending=False)
    best = json.loads(df.iloc[0].T.to_json())
    best['template_types'] = [best['template_types']]
    accuracy = best.pop('accuracy')
    pprint("{}: {}".format(data, accuracy))
    return best

configs = {}
ex_configs = None
optimal_configs = {i: get_best_config(i) for i in ['sat_package_v3', 'u2_raw', 'u4_raw']}
for k, v in optimal_configs.items():
    if ex_configs is None:
        ex_configs = {
            i: safe_open(i, v.keys()) for i in glob('./experiments/ppl_pmi_grid/results/outputs/*/config.json')}
    same_config = list(filter(lambda x: x[1] == v, ex_configs.items()))
    pprint(same_config)
    # print("find {} match".format(len(same_config)))
    configs[k] = os.path.dirname(same_config[0][0])

u4_opt = optimal_configs['u4_raw']
u4_opt['path_to_data'] = './data/sat_package_v3.jsonl'
same_config = list(filter(lambda x: x[1] == u4_opt, ex_configs.items()))
pprint(same_config)
configs['sat_package_v3/u4_raw'] = os.path.dirname(same_config[0][0])

u4_opt['path_to_data'] = './data/u2_raw.jsonl'
same_config = list(filter(lambda x: x[1] == u4_opt, ex_configs.items()))
pprint(same_config)
configs['u2_raw/u4_raw'] = os.path.dirname(same_config[0][0])
pprint()
pprint(configs)

