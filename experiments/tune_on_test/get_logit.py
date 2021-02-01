import alm

all_templates = ['is-to-what', 'is-to-as', 'rel-same', 'what-is-to', 'she-to-as', 'as-what-same']
data = ['sat']
models = [('roberta-large', 32, 512)]

for _model, _max_length, _batch in models:
    scorer = alm.RelationScorer(model=_model, max_length=_max_length)
    for _data in data:
        for _temp in all_templates:
            scorer.analogy_test(
                test=True,
                scoring_method='ppl_pmi',
                data=_data,
                template_type=_temp,
                batch_size=_batch,
                negative_permutation=True,
                skip_scoring_prediction=True
            )
            scorer.release_cache()


