import logging
import os
from itertools import product
from pprint import pprint
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
import alm
import pandas as pd

data = ['sat', 'u2', 'u4', 'google', 'bats']
model = ['roberta-large', 'gpt2-xl', 'bert-large-cased']
df = alm.get_report(export_prefix='main1')
df = df[df.scoring_method != 'ppl_head_masked']
df = df[df.scoring_method != 'ppl_tail_masked']
df = df[df.scoring_method != 'ppl_add_masked']
df = df[df.scoring_method != 'ppl_pmi']
g = df.groupby(['data', 'scoring_method']).accuracy.max()
g = g.to_frame()
g.reset_index(inplace=True)

df2 = alm.get_report(export_prefix='main2')
g2 = df2.groupby(['data', 'scoring_method']).accuracy.max()
g2 = g2.to_frame()
g2.reset_index(inplace=True)


def rename(x):
    if x == 'embedding_similarity':
        return 'embedding'
    if x == 'pmi':
        return 'Mask PMI'
    if x == 'ppl':
        return 'perplexity'
    if x == 'ppl_pmi':
        return 'PMI'
    else:
        raise ValueError('OMG')


df_full = pd.concat([g, g2])
df_full.reset_index()
df_full['accuracy'] = df_full['accuracy'] * 100
df_full['scoring_method'] = df_full['scoring_method'].apply(
    lambda x: rename(x))
df_full.sort_values(by=['data', 'scoring_method'])
df_full.to_csv('./experiments_results/summary/scoring_method.csv')

