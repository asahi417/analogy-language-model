import os
import json
from glob import glob
import pandas as pd

target_data = 'sat'

index = [
    'model', 'max_length', 'scoring_method', 'template_types', 'permutation_positive', 'permutation_negative',
    'aggregation_positive', 'aggregation_negative']
df = pd.DataFrame(index=index + ['accuracy'])

for i in glob('results/*'):
    if 'flatten' in i:
        continue
    with open(os.path.join(i, 'accuracy.json'), 'r') as f:
        accuracy = json.load(f)
    with open(os.path.join(i, 'config.json'), 'r') as f:
        config = json.load(f)
    path_to_data = config.pop('path_to_data')
    if target_data in path_to_data:
        df[len(df.T)] = [','.join(config[i]) if type(config[i]) is list else config[i] for i in index] +\
                        [round(accuracy['accuracy'] * 100, 2)]

df = df.T
df = df.sort_values(by=index, ignore_index=True)
print(df)
df.to_csv('results/summary.csv')
