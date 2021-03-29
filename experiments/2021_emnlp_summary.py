# Aggregate result for the paper
import pandas as pd

data = ['sat', 'u2', 'u4', 'google', 'bats']
models = [('roberta-large', 32, 512), ('gpt2-xl', 32, 256), ('bert-large-cased', 32, 1024)]
methods = ['ppl_marginal_bias', 'ppl_hypothesis_bias', 'ppl_based_pmi']
print('\n###############################')
print('## TUNED AP BIASED PPL SCORE ##')
print('###############################')

for d in data:
    df = pd.read_csv('./experiments_results/summary/experiment.ppl_variants.full.{}.csv'.format(d))
    print('DATASET: {}'.format(d))
    for method in methods:
        df_m = df[df.scoring_method == method]
        for model, _, _ in models:
            df_tmp = df_m[df_m.model == model]
            if len(df_tmp) == 0:
                continue
            df_tmp = df_tmp.sort_values(by=['accuracy_validation'], ascending=False)
            acc_val = list(df_tmp.head(1)['accuracy_validation'])[0]
            acc = df_tmp[df_tmp.accuracy_validation == acc_val].sort_values(by=['accuracy_test'])
            acc_test = list(acc['accuracy_test'])
            acc_test = acc_test[int(len(acc_test)/2)]
            acc_val = round(acc_val * 100, 1)
            acc_test = round(acc_test * 100, 1)
            acc_full = list(acc['accuracy'])[0]
            full_acc = round(acc_full * 100, 1)
            print('- {:<20} ({:<20}): {} (validation {}, full {})'.format(model, method, acc_test, acc_val, full_acc))


print('\n################################################')
print('## TUNED AP -PMI SCORE (TUNED ON FULL SET) ##')
print('################################################')
for d in data:
    df = pd.read_csv('./experiments_results/summary/experiment.ppl_variants.full.{}.csv'.format(d))
    print('DATASET: {}'.format(d))
    for method in methods:
        df_m = df[df.scoring_method == method]
        for model, _, _ in models:
            df_tmp = df_m[df_m.model == model]
            if len(df_tmp) == 0:
                continue
            df_tmp = df_tmp.sort_values(by=['accuracy'], ascending=False)
            acc = list(df_tmp.head(1)['accuracy'])[0]
            acc = round(acc * 100, 1)
            print('- {:<20} ({:<20}): {}'.format(model, method, acc))
