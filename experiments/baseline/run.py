import alm

all_templates = [['is-to-what'], ['is-to-as'], ['rel-same'], ['what-is-to'], ['she-to-as'], ['as-what-same']]
data = ['./data/sat_package_v3.jsonl', './data/u2.jsonl', './data/u4.jsonl']
lm = [('roberta-large', 32, 512), ('gpt2-xl', 32, 512)]
export_dir = './experiments/baseline/results'

for _model, _max_length, _batch in lm:
    scorer = alm.RelationScorer(model=_model, max_length=_max_length)
    for _data in data:
        for _temp in all_templates:

            def run(scoring_method):
                scorer.analogy_test(
                    scoring_method=scoring_method,
                    path_to_data=_data,
                    template_types=_temp,
                    batch_size=_batch,
                    export_dir=export_dir,
                    permutation_negative=False,
                    skip_scoring_prediction=True
                )
                scorer.release_cache()

            run('ppl')
            run('embedding_similarity')
            run('ppl_pmi')
            if 'gpt' not in _model:
                run('pmi')
