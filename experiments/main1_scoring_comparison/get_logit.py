import alm

all_templates = ['is-to-what', 'is-to-as', 'rel-same', 'what-is-to', 'she-to-as', 'as-what-same']
methods = ['pmi_feldman', 'embedding_similarity', 'ppl', 'ppl_based_pmi', 'ppl_head_masked', 'ppl_tail_masked']
data = ['sat', 'u2', 'u4', 'google', 'bats']
models = [('roberta-large', 32, 512), ('gpt2-xl', 32, 128), ('bert-large-cased', 32, 1024)]


for _model, _max_length, _batch in models:
    for scoring_method in methods:
        if 'gpt' in _model and scoring_method in ['ppl', 'ppl_based_pmi']:
            continue
        scorer = alm.RelationScorer(model=_model, max_length=_max_length)
        for _data in data:
            for _temp in all_templates:
                scorer.analogy_test(
                    scoring_method=scoring_method,
                    data=_data,
                    template_type=_temp,
                    batch_size=_batch,
                    skip_scoring_prediction=True)
                scorer.release_cache()

