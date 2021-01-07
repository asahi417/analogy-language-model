import json
import pandas as pd

data = ['data/sample.jsonl', 'data/sat_package_v3.jsonl', 'data/u2.jsonl', 'data/u4.jsonl']

for i in data:
    with open(i, 'r') as f:
        jsons = list(map(lambda x: json.loads(x), f.read().split('\n')))
        stem_a = list(map(lambda x: x['stem'][0], jsons))
        stem_b = list(map(lambda x: x['stem'][1], jsons))

        option_n = len(jsons[0]['choice'])
        opt = [
            stem_a,
            stem_b,
            list(map(
                lambda x: map(lambda y: y, x['choice'][y][x][0], jsons),
                range(option_n))),
            list(map(
                lambda x: map(lambda y: y, x['choice'][y][x][1], jsons),
                range(option_n)))
        ]
        print(opt)
        input()

        # for _i in :
        #     c_a = list(map(lambda x: x['choice'][_i][0], jsons))
        #     c_b = list(map(lambda x: x['choice'][_i][1], jsons))
        #     tmp = pd.DataFrame(jsons)
        #     print(tmp)
        #     input()
