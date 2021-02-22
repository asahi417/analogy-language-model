import alm

all_templates = ['is-to-what', 'is-to-as', 'rel-same', 'what-is-to', 'she-to-as', 'as-what-same']
data = ['sat', 'u2', 'u4', 'google', 'bats']
# data_test = ['sat']
data_test = ['sat', 'u2', 'u4', 'google', 'bats']
models = [('roberta-large', 32, 512), ('gpt2-xl', 32, 256), ('bert-large-cased', 32, 1024)]


for _model, _max_length, _batch in models:
    scorer = alm.RelationScorer(model=_model, max_length=_max_length)
    for _data in data:
        for _temp in all_templates:
            scorer.analogy_test(
                scoring_method='ppl_pmi',
                data=_data,
                template_type=_temp,
                batch_size=_batch,
                negative_permutation=True,
                skip_scoring_prediction=True
            )
            scorer.release_cache()
            if _data in data_test:
                scorer.analogy_test(
                    scoring_method='ppl_pmi',
                    data=_data,
                    template_type=_temp,
                    batch_size=_batch,
                    negative_permutation=True,
                    skip_scoring_prediction=True,
                    test=True
                )
                scorer.release_cache()

