# Aggregate result for the paper
import pandas as pd

print('\n#################################')
print('## SCORING FUNCTION COMPARISON ##')
print('#################################')
data = ['sat', 'u2', 'u4', 'google', 'bats']
models = [('roberta-large', 32, 512), ('gpt2-xl', 32, 256), ('bert-large-cased', 32, 1024)]
scoring_method = ['ppl_hypothesis_bias', 'ppl_marginal_bias', 'ppl_based_pmi']
methods = ['pmi_feldman', 'embedding_similarity', 'ppl']
df = pd.read_csv('experiments_results/summary/experiment.scoring_comparison.test.csv')
for d in data:
    df_tmp = df[df.data == d]
    print('DATASET: {}'.format(d))
    for model, _, _ in models:
        print('- {}'.format(model))
        df_tmp_tmp = df_tmp[df_tmp.model == model]
        for method in methods:
            df_tmp_tmp_tmp = df_tmp_tmp[df_tmp_tmp.scoring_method == method]
            if len(set(list(df_tmp_tmp_tmp['accuracy_validation']))) == 1:
                acc_val = round(list(df_tmp_tmp_tmp['accuracy_validation'])[0] * 100, 1)
                acc = [round(i * 100, 1) for i in list(df_tmp_tmp_tmp['accuracy'])]
                acc = acc[int(len(acc)/2)]
                print('\t - {:<20}: {} (validation {})'.format(method, acc, acc_val))

print('\n###########################')
print('## DEFAULT PPL BASED PMI ##')
print('###########################')
df_test = pd.read_csv('experiments_results/summary/experiment.scoring_comparison.default.test.csv')
df_val = pd.read_csv('experiments_results/summary/experiment.scoring_comparison.default.valid.csv')
for d in data:
    df_test_tmp = df_test[df_test.data == d]
    df_val_tmp = df_val[df_val.data == d]
    print('DATASET: {}'.format(d))
    for model, _, _ in models:
        df_val_tmp_tmp = df_val_tmp[df_val_tmp.model == model]
        df_test_tmp_tmp = df_test_tmp[df_test_tmp.model == model]

        if len(set(list(df_val_tmp_tmp['accuracy']))) == 1:
            acc_val = round(list(df_val_tmp_tmp['accuracy'])[0] * 100, 1)
            acc_test = round(list(df_test_tmp_tmp['accuracy'])[0] * 100, 1)
            if d == 'sat':
                full_acc = round((37 * acc_val + 337 * acc_test)/374, 1)
                print('- {:<20}: {} (validation {}, full {})'.format(model, acc_test, acc_val, full_acc))
            else:
                print('- {:<20}: {} (validation {})'.format(model, acc_test, acc_val))

print('\n#############################')
print('## WORD EMBEDDING BASELINE ##')
print('#############################')
df_test = pd.read_csv('./experiments_results/summary/experiment.word_embedding.test.csv')
df_val = pd.read_csv('./experiments_results/summary/experiment.word_embedding.valid.csv')
df_test_pmi = pd.read_csv('./experiments_results/summary/experiment.pmi.test.csv')
df_valid_pmi = pd.read_csv('./experiments_results/summary/experiment.pmi.valid.csv')
models = ['w2v', 'fasttext', 'glove']
for d in data:
    df_test_tmp = df_test[df_test.data == d]
    df_val_tmp = df_val[df_val.data == d]
    pmi_val = round(list(df_valid_pmi[df_valid_pmi.data == d]['pmi'])[0] * 100, 1)
    pmi_test = round(list(df_test_pmi[df_test_pmi.data == d]['pmi'])[0] * 100, 1)
    print('DATASET: {}'.format(d))
    if d == 'sat':
        full_acc = round((37 * pmi_val + 337 * pmi_test) / 374, 1)
        print('- {:<20}: {} (validation {}, full {})'.format('pmi', pmi_test, pmi_val, full_acc))
    else:
        print('- {:<20}: {} (validation {})'.format('pmi', pmi_test, pmi_val))
    for model in models:
        acc_val = round(float(df_val_tmp[model]) * 100, 1)
        acc_test = round(float(df_test_tmp[model]) * 100, 1)
        if d == 'sat':
            full_acc = round((37 * acc_val + 337 * acc_test) / 374, 1)
            print('- {:<20}: {} (validation {}, full {})'.format(model, acc_test, acc_val, full_acc))
        else:
            print('- {:<20}: {} (validation {})'.format(model, acc_test, acc_val))

print('\n############################')
print('## TUNED AP PPL-PMI SCORE ##')
print('############################')
models = [('roberta-large', 32, 512), ('gpt2-xl', 32, 256), ('bert-large-cased', 32, 1024)]
for d in data:
    df = pd.read_csv('./experiments_results/summary/experiment.ppl_variants.full.{}.csv'.format(d))
    df = df[df.scoring_method == 'ppl_based_pmi']
    print('DATASET: {}'.format(d))
    for model, _, _ in models:
        df_tmp = df[df.model == model]
        df_tmp = df_tmp.sort_values(by=['accuracy_validation'], ascending=False)
        acc_val = list(df_tmp.head(1)['accuracy_validation'])[0]
        acc = df_tmp[df_tmp.accuracy_validation == acc_val].sort_values(by=['accuracy_test'])
        acc_test = list(acc['accuracy_test'])
        acc_test = acc_test[int(len(acc_test)/2)]

        acc_val = round(acc_val * 100, 1)
        acc_test = round(acc_test * 100, 1)
        acc_full = list(acc['accuracy'])[0]
        full_acc = round(acc_full * 100, 1)
        print('- {:<20}: {} (validation {}, full {})'.format(model, acc_test, acc_val, full_acc))

print('\n################################################')
print('## TUNED AP PPL-PMI SCORE (TUNED ON FULL SET) ##')
print('################################################')
models = [('roberta-large', 32, 512), ('gpt2-xl', 32, 256), ('bert-large-cased', 32, 1024)]
for d in data:
    df = pd.read_csv('./experiments_results/summary/experiment.ppl_variants.full.{}.csv'.format(d))
    df = df[df.scoring_method == 'ppl_based_pmi']
    print('DATASET: {}'.format(d))
    for model, _, _ in models:
        df_tmp = df[df.model == model]
        df_tmp = df_tmp.sort_values(by=['accuracy'], ascending=False)
        acc = list(df_tmp.head(1)['accuracy'])[0]

        acc = round(acc * 100, 1)
        print('- {:<20}: {}'.format(model, acc))
