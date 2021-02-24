import json
from glob import glob

for i in glob('experiments_results/logit/*/*/ppl_based_pmi/*/config.json'):
    print(i)
    config = json.load(open(i))
    config['scoring_method'] = 'ppl_based_pmi'
    with open(i, 'w') as f:
        json.dump(config, f)
