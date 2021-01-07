import json
import pandas as pd

data = ['data/sat_package_v3.jsonl', 'data/u2.jsonl', 'data/u4.jsonl']

for i in data:
    with open(i, 'r') as f:
        jsons = list(map(lambda x: json.loads(x), f.read().split('\n')))
        stem_a = list(map(lambda x: x['stem'][0], jsons))
        stem_b = list(map(lambda x: x['stem'][1], jsons))
        for i in range(len(jsons[0]['choice'])):
        c_a = list(map(lambda x: x['choice'][0][0], jsons))
        c_b = list(map(lambda x: x['choice'][0][1], jsons))
        tmp = pd.DataFrame(jsons)
        print(tmp)
        input()
