import alm

data = ['sat', 'u2', 'u4', 'google', 'bats']
export_prefix = 'main2'
df = alm.get_report(export_prefix=export_prefix)

for i in data:
    tmp_df = df[df.data == i]
    tmp_df = tmp_df.sort_values(by='accuracy').head(0)
    print(tmp_df)
