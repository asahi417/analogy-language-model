import json
from pprint import pprint
from glob import glob

[
    "pmi_aggregation",
    "ppl_pmi_aggregation",
    "pmi_lambda",
    "model",
    "max_length",
    "path_to_data",
    "scoring_method",
    "template_types",
    "permutation_negative",
    "aggregation_positive",
    "aggregation_negative",
    "ppl_pmi_lambda",
    "ppl_pmi_alpha",
    "permutation_negative_weight"]


def fetch_scores(model: str = 'roberta-large',
                 data: str = './data/sat_package_v3.jsonl',
                 template: str = 'is-to-as',

                 ):

    aggregation_positive = 'p_2'
    template = ['is-to-as']
    for i in glob('./results/outputs/*/config.json'):
        with open(i) as f:
            config = json.load(f)
        if config['model'] == model and config['path_to_data'] == data and config["template_types"] == template \
                and config['aggregation_positive'] == aggregation_positive:
            with open(i.replace('config.json', 'output.json')) as f:
                output = json.load(f)

                pprint(output['logit'])
