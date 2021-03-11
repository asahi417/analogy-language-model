import os
import alm

import pandas as pd

os.makedirs('./experiments_results/summary', exist_ok=True)
data = ['sat', 'u2', 'u4', 'google', 'bats']


line_table = []
for i in data:
    table = {'data': i}
    val, test = alm.data_analogy.get_dataset_raw(i)
    table['size (validation|test)'] = '{}|{}'.format(len(val), len(test))
    table['candidate number'] = ','.join(list(set([str(len(i['choice'])) for i in test]
                                                  + [str(len(i['choice'])) for i in val])))
    if i == 'sat':
        table['category number'] = 2
    else:
        table['category number'] = len(list(set([i['prefix'] for i in test] + [i['prefix'] for i in val])))
    table['random expectation (test)'] = sum([1/len(i['choice']) for i in test])/len(test) * 100
    line_table.append(table)

print(pd.DataFrame(line_table))
pd.DataFrame(line_table).to_csv('experiments_results/summary/data.csv')


