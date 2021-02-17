import alm

export_prefix = 'main2.ppl_pmi'
df_val = alm.get_report(export_prefix=export_prefix)
df_val = df_val[df_val.data == 'sat']
df_test = alm.get_report(export_prefix=export_prefix, test=True)
df_test = df_test[df_test.data == 'sat']

df_val = df_val.sort_values(by=list(df_val.columns))
df_test = df_test.sort_values(by=list(df_val.columns))

accuracy_val = df_val.pop('accuracy').to_numpy()
accuracy_test = df_test.pop('accuracy').to_numpy()
assert df_val.shape == df_test.shape

df_test['accuracy_validation'] = accuracy_val
df_test['accuracy_test'] = accuracy_test

df_test['accuracy'] = (accuracy_val * 37 + accuracy_test * 337)/(37 + 337)
df_test = df_test.sort_values(by=['accuracy'], ascending=False)
df_test.to_csv('./experiments_results/summary/{}.csv'.format(export_prefix))
print(df_test['accuracy'].head(10))

