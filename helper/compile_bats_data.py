import os
import json
from random import randint, shuffle, seed
from glob import glob

##################
# Format Dataset #
##################
if not os.path.exists('./cache'):
    os.makedirs('./cache')
path_to_file = './cache/BATS_3.0'

if not os.path.exists(path_to_file):
    raise ValueError('download BATS dataset from \n'
                     'https://u.pcloud.link/publink/show?code=XZOn0J7Z8fzFMt7Tw1mGS6uI1SYfCfTyJQTV\n'
                     'unzip, and locate it at {}'.format(path_to_file))
dict_data = {}
for d in glob('{}/*'.format(path_to_file)):
    if not os.path.isdir(d):
        continue
    dict_data[d] = {}
    for t in glob('{}/*.txt'.format(d)):
        relation = os.path.basename(t)
        with open(t, 'r') as f:
            rel = list(map(
                lambda x: [i.split('/')[0] for i in x.split('\t')],
                filter(lambda y: len(y) > 0, f.read().split('\n'))
            ))
            rel = list(filter(lambda x: x[0] != x[1], rel))
            dict_data[d][relation] = rel

print('STATISTICS')
n = 0
for k, v in dict_data.items():
    for k_, v_ in v.items():
        print('- relation `{}/{}`: {}'.format(k, k_, len(v_)))
        n += len(v_)
print('total: {}'.format(n))

####################
# Create Negatives #
####################
seed(1234)
analogy_data = []
all_relation = []
for k, v in dict_data.items():
    for k_, v_ in v.items():
        if '4_Lexicographic_semantics' in k:
            _type = k_.split('[')[-1].split(' -')[0]
            negative_low = list(filter(lambda x: x != k_ and _type not in x, v.keys()))
        else:
            negative_low = list(filter(lambda x: x != k_, v.keys()))
        for stem in v_:
            if stem[0] == stem[1]:
                continue
            relation_list = list(filter(lambda x: x != stem and x[0] not in stem and x[1] not in stem, v_))
            label = relation_list[randint(0, len(relation_list) - 1)]
            if label[0] == label[1]:
                continue

            # pick up two word randomly from the first word in same relation group
            neg_1 = list(filter(lambda x: x not in stem and x not in label, map(lambda y: y[0], relation_list)))
            neg_1 = list(set(neg_1))
            neg_1 = [neg_1[randint(0, int(len(neg_1) / 2) - 1)], neg_1[randint(int(len(neg_1) / 2), len(neg_1) - 1)]]

            # pick up two word randomly from the second word in same relation group
            neg_2 = list(filter(lambda x: x not in stem and x not in label, map(lambda y: y[0], relation_list)))
            neg_2 = list(set(neg_2))
            neg_2 = [neg_2[randint(0, int(len(neg_2) / 2) - 1)], neg_2[randint(int(len(neg_2) / 2), len(neg_2) - 1)]]

            # pick up two word randomly from the second word in same relation group (low level)
            r_neg_type_l = negative_low[randint(0, len(negative_low) - 1)]
            r_neg_list_l = v[r_neg_type_l]
            neg_3 = r_neg_list_l[randint(0, len(r_neg_list_l) - 1)]

            choice = [label, neg_1, neg_2, neg_3]
            shuffle(choice)
            answer_id = choice.index(label)
            analogy_data.append({
                'stem': stem,
                "answer": answer_id,
                "choice": choice,
                "prefix": '{}/{}'.format(k, k_)  # add relation type as a meta information
            })
            all_relation.append('{}/{}'.format(k, k_))


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

os.makedirs('./data/bats', exist_ok=True)
with open('./data/bats/valid.jsonl', 'w') as writer:
    writer.write('\n'.join([json.dumps(d) for d in val]))
with open('./data/bats/test.jsonl', 'w') as writer:
    writer.write('\n'.join([json.dumps(d) for d in test]))
