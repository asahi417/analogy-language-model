import shutil
import json
import os
from glob import glob

for d in glob('./experiments/baseline/results/flatten_scores/*'):
    i = '{}/config.json'.format(d)
    if not os.path.exists(i):
        continue
    with open(i, 'r') as f:
        config = json.load(f)
    scoring_method = config['scoring_method'].replace('_', '-')
    model = config['model']
    data = config['path_to_data'].split('/')[-1].replace('.jsonl', '').replace('_', '-')
    template_types = config['template_types'][0]
    base_name = d.split('/')[-1]
    folder_name = '{}_{}_{}_{}_{}'.format(scoring_method, model, data, template_types, base_name)

    print('move {} to {}'.format(d, './experiments/baseline/results/flatten_scores/{}'.format(folder_name)))
    shutil.move(d, './experiments/baseline/results/flatten_scores/{}'.format(folder_name))
