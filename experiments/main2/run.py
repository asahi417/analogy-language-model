import alm

all_templates = ['is-to-what', 'is-to-as', 'rel-same', 'what-is-to', 'she-to-as', 'as-what-same']
methods = ['ppl_pmi']
data = ['sat', 'u2', 'u4', 'google', 'bats']
models = [('roberta-large', 32, 512), ('gpt2-xl', 32, 512)]


for _model, _max_length, _batch in models:
    for scoring_method in methods:
        if scoring_method in ['pmi', 'ppl_tail_masked', 'ppl_head_masked', 'ppl_add_masked'] and 'gpt' in _model:
            continue
        scorer = alm.RelationScorer(model=_model, max_length=_max_length)
        for _data in data:
            for _temp in all_templates:
                scorer.analogy_test(
                    scoring_method=scoring_method,
                    data=_data,
                    template_type=_temp,
                    batch_size=_batch,
                    negative_permutation=True,
                    skip_scoring_prediction=True
                )
                scorer.release_cache()

