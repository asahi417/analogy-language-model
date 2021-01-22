import alm
import os
import json
from glob import glob
import pandas as pd

all_templates = ['is-to-what', 'is-to-as', 'rel-same', 'what-is-to', 'she-to-as', 'as-what-same']
methods = ['ppl_tail_masked', 'ppl_head_masked', 'ppl', 'embedding_similarity', 'ppl_pmi', 'pmi', 'ppl_add_masked']
data = ['sat', 'u2', 'u4', 'google', 'bats']
models = [('roberta-large', 32, 512), ('gpt2-xl', 32, 512), ('bert-large-cased', 64, 512), ('gpt2-large', 32, 512)]

positive_permutation_aggregation = ['max', 'mean', 'min', 'p_0', 'p_1', 'p_2', 'p_3', 'p_4', 'p_5', 'p_6', 'p_7']
pmi_aggregation = ['max', 'mean', 'min', 'p_0', 'p_1', 'p_2', 'p_3', 'p_4', 'p_5', 'p_6', 'p_7', 'p_8', 'p_9', 'p_10',
                   'p_11']
ppl_pmi_aggregation = ['max', 'mean', 'min', 'p_0', 'p_1']
export_prefix = 'main'

for _model, _max_length, _batch in models:
    for scoring_method in methods:
        if scoring_method in ['pmi'] and 'gpt' in _model:
            continue
        scorer = alm.RelationScorer(model=_model, max_length=_max_length)
        for _data in data:
            for _temp in all_templates:
                shared = dict(
                    scoring_method=scoring_method,
                    data=_data,
                    template_type=_temp,
                    batch_size=_batch,
                    export_prefix=export_prefix,
                    no_inference=True,
                    positive_permutation_aggregation=positive_permutation_aggregation
                )
                if scoring_method == 'ppl_pmi':
                    shared['ppl_pmi_aggregation'] = ppl_pmi_aggregation
                if scoring_method == 'pmi':
                    shared['pmi_aggregation'] = pmi_aggregation
                scorer.analogy_test(**shared)
                scorer.release_cache()

# export as a csv
index = ['model', 'path_to_data', 'scoring_method', 'template_types', 'aggregation_positive', 'pmi_aggregation',
         'pmi_lambda', 'ppl_pmi_lambda', 'ppl_pmi_aggregation']
df = pd.DataFrame(index=index + ['accuracy'])

for i in glob('{}/outputs/*'.format(export_dir)):
    with open(os.path.join(i, 'accuracy.json'), 'r') as f:
        accuracy = json.load(f)
    with open(os.path.join(i, 'config.json'), 'r') as f:
        config = json.load(f)
    df[len(df.T)] = [','.join(config[i]) if type(config[i]) is list else config[i] for i in index] + \
                    [round(accuracy['accuracy'] * 100, 2)]

df = df.T
df = df.sort_values(by=index, ignore_index=True)
df.to_csv('{}/summary.csv'.format(export_dir))

