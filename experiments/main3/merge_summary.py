import alm

df_val = alm.get_report(export_prefix='main2')
df_val = df_val[df_val.data == 'sat']
df_test = alm.get_report(export_prefix='main3', test=True)

df_val = df_val.sort_values(by=list(df_val.columns)).reindex()
df_test = df_test.sort_values(by=list(df_val.columns)).reindex()

accuracy_val = df_val.pop('accuracy')
accuracy_test = df_test.pop('accuracy')
assert df_val.shape == df_test.shape

df_test['accuracy_validation'] = accuracy_val
df_test['accuracy_test'] = accuracy_test

df_test['accuracy'] = (accuracy_val * 37 + accuracy_test * 337)/(37 + 337)
df_test.to_csv('./experiments_results/summary/main3.combined.csv')
