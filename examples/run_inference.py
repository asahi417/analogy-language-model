""" Run model inference for an analogy dataset """
import alm

for i in range(-10, 10):
    scorer = alm.RelationScorer(model='roberta-large', max_length=32)
    scorer.analogy_test(
        path_to_data='./data/sat_package_v3.jsonl',
        template_types=['rel-same'],
        batch_size=512,
        export_dir='./results_pmi_tuning',
        permutation_negative=False,
        skip_scoring_prediction=True,
        pmi_lambda=i*0.1
    )
    scorer.release_cache()
