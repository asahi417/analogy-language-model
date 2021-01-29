import logging
import os
from itertools import product
from pprint import pprint
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
import alm
import pandas as pd

data = ['sat', 'u2', 'u4', 'google', 'bats']
model = ['roberta-large', 'gpt2-xl', 'bert-large-cased']
export_prefix = 'main2'
df = alm.get_report(export_prefix=export_prefix)
os.makedirs('./experiments_results/summary/main2_summary')
print('ALPHA/ETA')
group = df.groupby(['data', 'model']).accuracy.max()
group_a = df.groupby(['data', 'model', 'ppl_pmi_alpha']).accuracy.max()
group_n = df.groupby(['data', 'model', 'negative_permutation_weight']).accuracy.max()
group_na = df.groupby(['data', 'model', 'ppl_pmi_alpha', 'negative_permutation_weight']).accuracy.max()
output = {i: {} for i in data}
for i, _model in product(data, model):
    output[i][_model] = {
        'ppl_neg_alpha': group[i][_model] * 100,
        'ppl_neg': group_a[i][_model][0.0] * 100,
        'ppl_alpha': group_n[i][_model][0.0] * 100,
        'ppl': group_na[i][_model][0.0][0.0] * 100
    }
pprint(output)
df_ = pd.DataFrame(output)
pprint(df_)
df_.to_csv('./experiments_results/summary/main2_summary/alpha_eta.csv')

for p in ['template_type', 'ppl_pmi_aggregation', 'positive_permutation_aggregation', 'negative_permutation_aggregation']:
    print('\n{}'.format(p))
    group = df.groupby(['data', 'model', p]).accuracy.max()
    output = {i: {} for i in data}
    for i, _model in product(data, model):
        sub_group = group[i][_model]
        top3 = list(sub_group.sort_values(ascending=False)[:3].keys())
        output[i][_model] = ','.join(top3)
    df_ = pd.DataFrame(output)
    pprint(df_)
    df_.to_csv('./experiments_results/summary/main2_summary/{}.csv'.format(p))

