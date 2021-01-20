import os
import json
from random import randint, shuffle, seed

##################
# Format Dataset #
##################
if not os.path.exists('./cache'):
    os.makedirs('./cache')
path_to_file = './cache/questions-words.txt'
export_to_file = './data/google.jsonl'

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
            neg_1 = [neg_1[randint(0, int(len(neg_1)/2))], neg_1[randint(int(len(neg_1)/2), len(neg_1) - 1)]]

            # pick up two word randomly from the second word in same relation group
            neg_2 = list(filter(lambda x: x not in stem and x not in label,
                                map(lambda y: y[0][1], relation_list)))
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
print('Dataset built: {}, exporting to {}'.format(len(analogy_data), export_to_file))
with open(export_to_file, 'w') as writer:
    writer.write('\n'.join([json.dumps(d) for d in analogy_data]))
