import alm
import os
import json
from glob import glob
import pandas as pd

aggregation_positives = ['max', 'mean', 'min', 'p_0', 'p_1', 'p_2', 'p_3', 'p_4', 'p_5', 'p_6', 'p_7']
export_dir = './results_ppl_pmi_tuning'
# get accuracy
scorer = alm.RelationScorer(model='roberta-large', max_length=32)
for i in [-1.5, -1, 0.5, 0, 0.5, 1, 1.5]:
    for _i in [-1.5, -1, 0.5, 0, 0.5, 1, 1.5]:

        list(map(lambda x: scorer.analogy_test(
            scoring_method='ppl_pmi',
            path_to_data='./data/sat_package_v3.jsonl',
            template_types=['as-what-same'],
            aggregation_positive=x,
            ppl_pmi_lambda=i,
            ppl_pmi_alpha=_i,
            no_inference=True,
            overwrite_output=False,
            export_dir=export_dir
            ),  aggregation_positives))
        # list(map(lambda x: scorer.analogy_test(
        #     scoring_method='ppl_pmi',
        #     path_to_data='./data/sat_package_v3.jsonl',
        #     template_types=['as-what-same'],
        #     aggregation_positive=x,
        #     ppl_pmi_alpha=i,
        #     no_inference=True,
        #     overwrite_output=False,
        #     export_dir=export_dir
        # ), aggregation_positives))

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
