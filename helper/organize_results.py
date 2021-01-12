import shutil
import json
import os
from glob import glob

# for m in ['baseline', 'pmi_grid']:
for m in ['baseline']:
    for d in glob('./experiments/{}/results/flatten_scores/*'.format(m)):
        i = '{}/config.json'.format(d)
        if not os.path.exists(i):
            continue
        with open(i, 'r') as f:
            config = json.load(f)
        scoring_method = config['scoring_method'].replace('_', '-')
        model = config['model']
        data = config['path_to_data'].split('/')[-1].replace('.jsonl', '').replace('_', '-')
        template_types = config['template_types'][0]
        base_name = d.split('/')[-1].split('_')[-1]
        folder_name = '{}_{}_{}_{}_{}'.format(scoring_method, model, data, template_types, base_name)

        print('move {} to {}'.format(d, './experiments/{}/results/flatten_scores/{}'.format(m, folder_name)))
        shutil.move(d, './experiments/{}/results/flatten_scores/{}'.format(m, folder_name))
