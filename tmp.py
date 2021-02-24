import json
from glob import glob

for i in glob('experiments_results/logit/*/*/pmi_feldman/*/config.json'):
    print(i)
    config = json.load(open(i))
    config['scoring_method'] = 'pmi_feldman'
    with open(i, 'w') as f:
        json.dump(config, f)
