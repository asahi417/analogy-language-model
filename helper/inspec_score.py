""" Run scoring over all the baseline data """
import json
from pprint import pprint
from glob import glob

if __name__ == '__main__':
    model = 'roberta-large'
    data = 'sat'
    template = ['is-to-as']
    for i in glob('./results/outputs/*/config.json'):
        with open(i) as f:
            config = json.load(f)
        if config['model'] == model and config['path_to_data'] == data and config["template_types"] == template \
                and config['aggregation_positive'] == aggregation_positive:
            with open(i.replace('config.json', 'output.json')) as f:
                output = json.load(f)

                pprint(output['logit'])
