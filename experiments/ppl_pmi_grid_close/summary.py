import alm
import os
import json
from glob import glob
import pandas as pd

aggregation_positives = ['p_2']
export_dir = './experiments/ppl_pmi_grid/results'
# get accuracy
scorer = alm.RelationScorer(model='roberta-large', max_length=32)
for alpha in map(lambda x: x/10, range(10, 20)):
    for lam in map(lambda x: x / 10, range(-10, 0)):

        list(map(lambda x: scorer.analogy_test(
            scoring_method='ppl_pmi',
            path_to_data='./data/sat_package_v3.jsonl',
            template_types=['as-what-same'],
            aggregation_positive=x,
            ppl_pmi_lambda=lam,
            ppl_pmi_alpha=alpha,
            no_inference=True,
            overwrite_output=False,
            export_dir=export_dir
            ),  aggregation_positives))

# export as a csv
index = ['model', 'path_to_data', 'scoring_method', 'template_types', 'aggregation_positive', 'ppl_pmi_lambda', 'ppl_pmi_alpha']
df = pd.DataFrame(index=index + ['accuracy'])

for i in glob('./{}/outputs/*'.format(export_dir)):
    with open(os.path.join(i, 'accuracy.json'), 'r') as f:
        accuracy = json.load(f)
    with open(os.path.join(i, 'config.json'), 'r') as f:
        config = json.load(f)
    df[len(df.T)] = [','.join(config[i]) if type(config[i]) is list else config[i] for i in index] + \
                    [round(accuracy['accuracy'] * 100, 2)]

df = df.T
df = df.sort_values(by=index, ignore_index=True)
df.to_csv('./{}/summary.csv'.format(export_dir))
