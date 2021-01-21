import json
from pprint import pprint
from glob import glob

import pandas as pd


def safe_open(_file):
    with open(_file, 'r') as f:
        return json.load(f)


for i in ['sat_package_v3', 'u2_raw', 'u4_raw']:
    df = pd.read_csv('./experiments/ppl_pmi_grid/results/summary.{}.csv'.format(i), index_col=0)
    df = df.sort_values(by='accuracy', ascending=False)
    best = json.loads(df.iloc[0].T.to_json())
    accuracy = best.pop('accuracy')
    print("{}: {}".format(i, accuracy))
    ex_configs = {i: safe_open(i) for i in glob('./experiments/ppl_pmi_grid/results/outputs/*/config.json')}
    # check duplication
    same_config = list(filter(lambda x: x[1] == best, ex_configs.items()))
    print(same_config)

# [
#     "pmi_aggregation",
#     "ppl_pmi_aggregation",
#     "pmi_lambda",
#     "model",
#     "max_length",
#     "path_to_data",
#     "scoring_method",
#     "template_types",
#     "permutation_negative",
#     "aggregation_positive",
#     "aggregation_negative",
#     "ppl_pmi_lambda",
#     "ppl_pmi_alpha",
#     "permutation_negative_weight"]
