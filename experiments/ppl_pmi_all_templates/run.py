from itertools import combinations, chain
import alm

all_templates = ['is-to-what', 'is-to-as', 'rel-same', 'what-is-to', 'she-to-as', 'as-what-same']
export_dir = './results_ppl_pmi_all_templates'


all_comb = list(chain(*[list(combinations(all_templates, i)) for i in range(2, len(all_templates))]))


_model, _max_length, _batch = 'roberta-large', 32, 512
scorer = alm.RelationScorer(model=_model, max_length=_max_length)

for _temp in all_comb:
    scorer.analogy_test(
        scoring_method='ppl_pmi',
        path_to_data='./data/sat_package_v3.jsonl',
        template_types=_temp,
        batch_size=_batch,
        export_dir=export_dir,
        permutation_negative=False,
        skip_scoring_prediction=True
    )
    scorer.release_cache()
