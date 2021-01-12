import json
from glob import glob


def get_dataset(path_to_data: str):
    """ Get prompted SAT-type dataset: a list of (answer: int, prompts: list, stem: list, choice: list)"""
    with open(path_to_data, 'r') as f:
        return list(filter(None, map(lambda x: json.loads(x) if len(x) > 0 else None, f.read().split('\n'))))


def get_pmi(smooth, data):
    all_dict = {}
    for i in glob('./cache/ppmi/{}*smooth={}.csv'.format(data, smooth)):
        with open(i, 'r') as f:
            tmp = list(filter(lambda x: len(x) == 3, map(lambda x: x.split('\t'), f.read().split('\n'))))
        tmp_dict = {'{}-{}'.format(h, t): float(score) for h, t, score in tmp if score != 'x'}
        if len(set(tmp_dict.keys()).intersection(set(all_dict.keys()))) != 0:
            inter = set(tmp_dict.keys()).intersection(set(all_dict.keys()))
            for ii in inter:
                assert all_dict[ii] == tmp_dict[ii]
        all_dict.update(tmp_dict)
    return all_dict


def get_accuracy(dataset, pmi):
    accuracy = []
    none_index = []
    for n, i in enumerate(dataset):
        a = i['answer']
        score = [pmi['-'.join(c)] if '-'.join(c) in pmi.keys() else -100 for c in i['choice']]
        if all(s == -100 for s in score):
            none_index.append(n)
        else:
            accuracy.append(int(score.index(max(score)) == a))
    accuracy = sum(accuracy)/len(accuracy)
    print('non-pmi data: {}'.format(len(none_index)))
    print('accuracy: {}'.format(accuracy))
    return accuracy, none_index


def load_relative(path_to_relative):
    with open(path_to_relative, 'r') as f:
        tmp = [v.split(' ') for v in f.read().split('\n')[1:]]
        relative_dict = {v[0].replace('__', '-'): list(map(float, v)) for v in tmp}
        print(relative_dict)
        input()



if __name__ == '__main__':
    relatives = ['./cache/relative/relative_sat.txt', './cache/relative/relative_u2.txt', './cache/relative/relative_u4.txt']
    for r in relatives:
        load_relative(r)

    result = {}
    for path in ['./data/sat_package_v3.jsonl', './data/u2_raw.jsonl', './data/u4_raw.jsonl']:
        data = get_dataset(path)
        for smooth_ in [0.5, 1]:
            # for data_ in ['sat', 'u']:
            if 'sat' in path:
                pmi_dict = get_pmi(smooth_, 'sat')
            else:
                pmi_dict = get_pmi(smooth_, 'u')
            a, none_ind = get_accuracy(data, pmi_dict)
            result[len(result)] = {'smooth': smooth_, 'accuracy': a, 'path_to_data': path}
    print(result)
