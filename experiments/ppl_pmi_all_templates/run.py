import alm
# from itertools import combinations, chain

# all_templates = ['is-to-what', 'is-to-as', 'rel-same', 'what-is-to', 'she-to-as', 'as-what-same']
# all_comb = list(chain(*[list(combinations(all_templates, i)) for i in range(2, len(all_templates))]))
all_comb = [('is-to-what', 'is-to-as', 'rel-same', 'as-what-same'),
('is-to-what', 'is-to-as', 'what-is-to', 'she-to-as'),
('is-to-what', 'is-to-as', 'what-is-to', 'as-what-same'),
('is-to-what', 'is-to-as', 'she-to-as', 'as-what-same'),
('is-to-what', 'rel-same', 'what-is-to', 'she-to-as'),
('is-to-what', 'rel-same', 'what-is-to', 'as-what-same'),
('is-to-what', 'rel-same', 'she-to-as', 'as-what-same'),
('is-to-what', 'what-is-to', 'she-to-as', 'as-what-same'),
('is-to-as', 'rel-same', 'what-is-to', 'she-to-as'),
('is-to-as', 'rel-same', 'what-is-to', 'as-what-same'),
('is-to-as', 'rel-same', 'she-to-as', 'as-what-same'),
('is-to-as', 'what-is-to', 'she-to-as', 'as-what-same'),
('rel-same', 'what-is-to', 'she-to-as', 'as-what-same'),
('is-to-what', 'is-to-as', 'rel-same', 'what-is-to', 'she-to-as'),
('is-to-what', 'is-to-as', 'rel-same', 'what-is-to', 'as-what-same'),
('is-to-what', 'is-to-as', 'rel-same', 'she-to-as', 'as-what-same'),
('is-to-what', 'is-to-as', 'what-is-to', 'she-to-as', 'as-what-same'),
('is-to-what', 'rel-same', 'what-is-to', 'she-to-as', 'as-what-same'),
('is-to-as', 'rel-same', 'what-is-to', 'she-to-as', 'as-what-same')]


_model, _max_length, _batch = 'roberta-large', 32, 512
scorer = alm.RelationScorer(model=_model, max_length=_max_length)

for _temp in all_comb:
    scorer.analogy_test(
        scoring_method='ppl_pmi',
        path_to_data='./data/sat_package_v3.jsonl',
        template_types=_temp,
        batch_size=_batch,
        export_dir='./experiments/ppl_pmi_all_templates/results',
        permutation_negative=False,
        skip_scoring_prediction=True
    )
    scorer.release_cache()
