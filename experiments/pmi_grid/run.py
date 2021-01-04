import alm

for i in [-2.0, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2.0]:
    scorer = alm.RelationScorer(model='roberta-large', max_length=32)
    scorer.analogy_test(
        scoring_method='pmi',
        path_to_data='./data/sat_package_v3.jsonl',
        template_types=['rel-same'],
        batch_size=512,
        export_dir='./experiments/pmi_grid/results',
        permutation_negative=False,
        skip_scoring_prediction=True,
        pmi_lambda=i
    )
    scorer.release_cache()
