import alm
import os
import json
from glob import glob
import pandas as pd

pmi_aggregations = ['max', 'mean', 'min', 'p_0', 'p_1', 'p_2', 'p_3', 'p_4', 'p_5', 'p_6', 'p_7', 'p_8', 'p_9', 'p_10', 'p_11']
aggregation_positives = ['max', 'mean', 'min', 'p_0', 'p_1', 'p_2', 'p_3', 'p_4', 'p_5', 'p_6', 'p_7']

all_templates = [['is-to-what'], ['is-to-as'], ['rel-same'], ['what-is-to'], ['she-to-as'], ['as-what-same']]
data = ['./data/sat_package_v3.jsonl', './data/u2.jsonl', './data/u4.jsonl']
lm = [('roberta-large', 32, 512), ('gpt2-xl', 32, 512)]
export_dir = './results'

for _model, _max_length, _batch in lm:
    scorer = alm.RelationScorer(model=_model, max_length=_max_length)
    for _data in data:
        for _temp in all_templates:
            for aggregation_positive in aggregation_positives:

                def run(scoring_method, pmi_aggregation=None):
                    scorer.analogy_test(
                        scoring_method=scoring_method,
                        path_to_data=_data,
                        template_types=_temp,
                        batch_size=_batch,
                        export_dir=export_dir,
                        permutation_negative=False,
                        no_inference=True,
                        overwrite_output=True,
                        aggregation_positive=aggregation_positive,
                        pmi_aggregation=pmi_aggregation
                    )
                    scorer.release_cache()

                run('ppl')
                run('embedding_similarity')
                run('ppl_pmi')
                if 'gpt' not in _model:
                    for _pmi_aggregation in pmi_aggregations:
                        run('pmi', _pmi_aggregation)

# export as a csv
index = ['model', 'path_to_data', 'scoring_method', 'template_types', 'aggregation_positive', 'pmi_aggregation', 'pmi_lambda', 'ppl_pmi_lambda']
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
