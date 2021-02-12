import alm

df_val = alm.get_report(export_prefix='main2')
df_val = df_val[df_val.data == 'sat']
df_test = alm.get_report(export_prefix='main3', test=True)

df_val = df_val.sort_values(by=list(df_val.columns)).reindex()
df_test = df_test.sort_values(by=list(df_val.columns)).reindex()
accuracy_val = df_val.pop('accuracy')
accuracy_test = df_test.pop('accuracy')

print(df_val.shape)
print(df_test.shape)
check = (df_val != df_test).values.sum()
assert check == 0, '{}'.format(check)
df_test['accuracy_validation'] = accuracy_val
accuracy_test.to_csv('./experiments_results/summary/main3.test.csv', index_col=0)
