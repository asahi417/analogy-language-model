import logging
from itertools import product
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
import alm

data = ['sat', 'u2', 'u4', 'google', 'bats']
model = ['roberta-large', 'gpt2-xl', 'bert-large-cased']
export_prefix = 'main2'
df = alm.get_report(export_prefix=export_prefix)

for i, _model in product(data, model):
    tmp_df = df[df.data == i][df.model == _model]
    acc = tmp_df.sort_values(by='accuracy', ascending=False).head(1)['accuracy'].values[0]
    acc_neg = tmp_df[tmp_df['ppl_pmi_alpha'] == 0.0].sort_values(
        by='accuracy', ascending=False).head(1)['accuracy'].values[0]
    acc_pmi = tmp_df[tmp_df['negative_permutation_weight'] == 0.0].sort_values(
        by='accuracy', ascending=False).head(1)['accuracy'].values[0]
    acc_default = tmp_df[tmp_df['negative_permutation_weight'] == 0.0][tmp_df['ppl_pmi_alpha'] == 0.0].sort_values(
        by='accuracy', ascending=False).head(1)[
        'accuracy'].values[0]
    print('{}/{}'.format(i, _model))
    print(acc_default*100)
    print(acc_pmi * 100)
    print(acc_neg * 100)
    print(acc * 100)

# for i, _model in product(data, model):
#     for s in ['pmi', 'embedding_similarity', 'ppl_pmi']:
#         tmp_df = df[df.data == i][df.model == _model]
#         acc = tmp_df.sort_values(by='accuracy', ascending=False).head(1)['accuracy'].values[0]
#         acc_neg = tmp_df[tmp_df['ppl_pmi_alpha'] == 0.0].sort_values(
#             by='accuracy', ascending=False).head(1)['accuracy'].values[0]
#         acc_pmi = tmp_df[tmp_df['negative_permutation_weight'] == 0.0].sort_values(
#             by='accuracy', ascending=False).head(1)['accuracy'].values[0]
#         acc_default = tmp_df[tmp_df['negative_permutation_weight'] == 0.0][tmp_df['ppl_pmi_alpha'] == 0.0].sort_values(
#             by='accuracy', ascending=False).head(1)[
#             'accuracy'].values[0]
#         print('{}/{}'.format(i, _model))
#         print(acc_default*100)
#         print(acc_pmi * 100)
#         print(acc_neg * 100)
#         print(acc * 100)
#
#
#
