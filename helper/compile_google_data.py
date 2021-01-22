import os
import json
from random import randint, shuffle, seed
from copy import deepcopy


def check_validity(json_data):
    tmp = deepcopy(json_data)
    for t, i in enumerate(json_data):
        a = i['answer']
        choice = []
        for n, (c_h, c_t) in enumerate(i['choice']):
            if len(c_h) == 0 or len(c_t) == 0:
                raise ValueError(i)
            if c_h in i['stem'] or c_t in i['stem']:
                print("found duplication: {} \n {}".format((c_h, c_t), i))
                if a == n:
                    raise ValueError('answer would be dropped')
                if a > n:
                    a = a - 1
            else:
                choice.append((c_h, c_t))
        tmp[t]['answer'] = a
        tmp[t]['choice'] = choice
        assert tmp[t]['choice'][tmp[t]['answer']] == i['choice'][i['answer']], str((
                tmp[t]['choice'][tmp[t]['answer']], i['choice'][i['answer']]
        ))

    return tmp

##################
# Format Dataset #
##################
if not os.path.exists('./cache'):
    os.makedirs('./cache')
path_to_file = './cache/questions-words.txt'

if not os.path.exists(path_to_file):
    os.system('wget -O {} http://download.tensorflow.org/data/questions-words.txt'.format(path_to_file))

with open(path_to_file, 'r') as f:
    data = list(filter(lambda y: len(y) > 0, f.read().split('\n')))
dict_data = {}
relation = None
for d in data:
    if ':' in d:
        relation = d.replace(': ', '')
        dict_data[relation] = []
    else:
        q, w, e, r = d.split(' ')
        dict_data[relation].append([[q, w], [e, r]])

print('STATISTICS')
n = 0
for k, v in dict_data.items():
    print('- relation `{}`: {}'.format(k, len(v)))
    n += len(v)
print('total: {}'.format(n))

####################
# Create Negatives #
####################
relation_m = ['capital-common-countries', 'capital-world', 'currency', 'city-in-state', 'family']
relation_m_negative = {
    'capital-common-countries': ['family', 'currency'],
    'capital-world': ['family', 'currency'],
    'currency': ['capital-common-countries', 'capital-world', 'family', 'city-in-state'],
    'city-in-state': ['family', 'currency'],
    'family': ['capital-common-countries', 'capital-world', 'currency', 'city-in-state']
}
relation_s = ['gram1-adjective-to-adverb', 'gram2-opposite', 'gram3-comparative', 'gram4-superlative',
              'gram5-present-participle', 'gram6-nationality-adjective', 'gram7-past-tense', 'gram8-plural',
              'gram9-plural-verbs']

seed(1234)
analogy_data = []
all_relation = []
for is_mor in [False, True]:
    if is_mor:
        relation_type = relation_m
        relation_negative = relation_m_negative
    else:
        relation_type = relation_s
        relation_negative = {i: list(filter(lambda x: x != i, relation_s)) for i in relation_s}
    existing_stem = []
    for r in relation_type:
        relation_list = dict_data[r]
        for stem, label in relation_list:
            if stem in existing_stem:
                continue

            # pick up two word randomly from the first word in same relation group
            neg_1 = list(filter(lambda x: x not in stem and x not in label,
                                map(lambda y: y[0][0], relation_list)))
            neg_1 = list(set(neg_1))
            neg_1 = [neg_1[randint(0, int(len(neg_1)/2))], neg_1[randint(int(len(neg_1)/2), len(neg_1) - 1)]]

            # pick up two word randomly from the second word in same relation group
            neg_2 = list(filter(lambda x: x not in stem and x not in label,
                                map(lambda y: y[0][1], relation_list)))
            neg_2 = list(set(neg_2))
            neg_2 = [neg_2[randint(0, int(len(neg_2) / 2))], neg_2[randint(int(len(neg_2) / 2), len(neg_2) - 1)]]

            # pick up two word randomly from the second word in the other relation group
            rel_neg_type = relation_negative[r][randint(0, len(relation_negative[r]) - 1)]
            relation_negative_list = dict_data[rel_neg_type]

            neg_3 = relation_negative_list[randint(0, len(relation_negative_list) - 1)][0]
            choice = [label, neg_1, neg_2, neg_3]
            shuffle(choice)
            answer_id = choice.index(label)
            analogy_data.append({
                'stem': stem,
                "answer": answer_id,
                "choice": choice,
                "prefix": r  # add relation type as a meta information
            })
            existing_stem.append(stem)
            all_relation.append(r)

#########################
# validation/test split #
#########################
all_relation = list(set(all_relation))
val = []
test = []
ratio = 0.1
for r in all_relation:
    sub = list(filter(lambda x: x['prefix'] == r, analogy_data))
    val_size = int(len(sub) * ratio)
    print('- relation: {}, val size: {}'.format(r, val_size))
    shuffle(sub)
    val += sub[:val_size]
    test += sub[val_size:]
print('total     : {}'.format(len(analogy_data)))
print('validation: {}'.format(len(val)))
print('test      : {}'.format(len(test)))
print('Dataset built: {}'.format(len(analogy_data)))

os.makedirs('./data/google', exist_ok=True)
with open('./data/google/valid.jsonl', 'w') as writer:
    writer.write('\n'.join([json.dumps(d) for d in val]))
with open('./data/google/test.jsonl', 'w') as writer:
    writer.write('\n'.join([json.dumps(d) for d in test]))
