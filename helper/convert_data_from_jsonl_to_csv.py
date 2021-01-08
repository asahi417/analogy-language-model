""" convert data from jsonl to csv """
import json
import pandas as pd

data = ['data/sat_package_v3.jsonl', 'data/u2.jsonl', 'data/u4.jsonl']


# format preserved
for i in data:
    with open(i, 'r') as f:
        jsons = list(map(lambda x: json.loads(x), f.read().split('\n')))
        ans = list(map(lambda x: x['answer'], jsons))
        pre = list(map(lambda x: x['prefix'], jsons))
        stem_a = list(map(lambda x: x['stem'][0], jsons))
        stem_b = list(map(lambda x: x['stem'][1], jsons))

        option_n = max(map(lambda x: len(jsons[x]['choice']), range(len(jsons))))
        ind = ['stem_head', 'stem_tail']
        ind += list(map(lambda x: 'opt_{}_head'.format(x), range(option_n)))
        ind += list(map(lambda x: 'opt_{}_tail'.format(x), range(option_n)))
        ind += ['answer', 'file_prefix']
        opt = [stem_a, stem_b]

        opt += list(map(
            lambda x: list(map(lambda y: y['choice'][x][0] if len(y['choice']) > x else '', jsons)),
            range(option_n)))
        opt += list(map(
            lambda x: list(map(lambda y: y['choice'][x][1] if len(y['choice']) > x else '', jsons)),
            range(option_n)))
        opt += [ans, pre]
        tmp = pd.DataFrame(opt, index=ind)
        tmp = tmp.sort_index()
        tmp = tmp.T
        print(tmp.head())
        tmp.to_csv(i.replace('.jsonl', '.csv'))
