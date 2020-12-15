""" Run model inference for an analogy dataset """
import alm

scorer = alm.RelationScorer(model='roberta-large')
scorer.analogy_test(
    path_to_data='./data/sat_package_v3.jsonl',
    template_types=['is-to-what'],
    batch_size=16,
    aggregation_positive='mean',
    permutation_negative=True,
    aggregation_negative='mean',
    export_dir='./results')
