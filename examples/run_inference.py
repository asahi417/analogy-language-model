""" Run model inference for an analogy dataset """
import alm

scorer = alm.RelationScorer(model='roberta-large')
template_types = ['is-to-what']
path_to_data = './data/sat_package_v3.jsonl'
scorer.analogy_test(
    path_to_data=path_to_data,
    template_types=template_types,
    batch_size=16,
    permutation_positive=True,
    aggregation_positive='mean',
    permutation_negative=True,
    aggregation_negative='mean',
    export_dir='./results')
