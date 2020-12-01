import json
import pickle
import os
from glob import glob

partial_config = {}
target = 'sat_package_v3'
for i in glob('results_partial/*/config.json'):
    with open(i) as f:
        config = json.load(f)
    print(i, config)
    path_to_data = config.pop('path_to_data')
    if '{}-'.format(target) in path_to_data:
        num = int(path_to_data.split('{}-'.format(target))[-1].replace('.jsonl', ''))
        if len(partial_config) == 0:
            partial_config[str(len(partial_config))] = {
                'config': config, 'path': [(num, i.replace('/config.json', ''))]}
        else:
            flg = False
            for k, v in partial_config.items():
                if config == v['config']:
                    flg = True
                    break
            if flg:
                partial_config[k]['path'].append((num, i.replace('/config.json', '')))

            else:
                partial_config[str(len(partial_config))] = {
                    'config': config, 'path': [(num, i.replace('/config.json', ''))]}

for k, v in partial_config.items():
    path = list(list(zip(*sorted(v['path'], key=lambda x: x[0])))[1])
    path_n = list(list(zip(*sorted(v['path'], key=lambda x: x[0])))[0])
    config = v['config']
    if len(path) != 4:
        print()
        print(path)
        print(path_n)
        print(config)
        print()
        continue

    print(path, path_n)


    new_checkpoint = 'flatten_{}_{}'.format(target, '_'.join(config['template_types']))
    if config['permutation_positive']:
        new_checkpoint += '_pos'
    if config['permutation_negative']:
        new_checkpoint += '_neg'
    new_checkpoint += '_' + '-'.join([i.replace('results_partial/', '') for i in path])
    print(new_checkpoint)
    print(config)
    config['path_to_data'] = './data/sat_package_v3.jsonl'
    for o in ["aggregation_positive", "aggregation_negative"]:
        config[o] = 'n/a'
    flatten_score_concat = []
    for p in path:
        with open(os.path.join(p, 'flatten_score.pkl'), "rb") as fp:  # Unpickling
            flatten_score_concat += pickle.load(fp)

    os.makedirs(os.path.join('results', new_checkpoint), exist_ok=True)
    with open(os.path.join('results', new_checkpoint, 'flatten_score.pkl'), "wb") as fp:
        pickle.dump(flatten_score_concat, fp)
    with open(os.path.join('results', new_checkpoint, 'config.json'), 'w') as f:
        json.dump(config, f)

    for i in path:
        os.rename(i, i + '_done')
