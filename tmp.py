import json
from glob import glob

for i in glob('./experiments_results/logit/*/*/*/*/config.json'):
    with open(i, 'r') as f:
        c = json.load(f)
    c['max_length'] = 32
    with open(i, 'w') as f:
        json.dump(c, f)