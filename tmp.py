import json
from glob import glob

for i in glob('experiments_results/logit/*/*/ppl_feldman/*/config.json'):
    config = json.load(open(i))
    config['scoring_method'] = 'pmi_feldman'
    with open(i, 'w') as f:
        json.dump(config, f)
