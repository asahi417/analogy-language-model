import alm

all_templates = ['is-to-what', 'is-to-as', 'rel-same', 'what-is-to', 'she-to-as', 'as-what-same']
data = ['sat', 'u2', 'u4', 'google', 'bats']
models = [('roberta-large', 32, 512), ('gpt2-xl', 32, 128), ('bert-large-cased', 32, 1024)]
methods = ['ppl_tail_masked', 'ppl_head_masked', 'ppl', 'embedding_similarity', 'ppl_pmi', 'ppl_add_masked', 'pmi']

positive_permutation_aggregation = ['max', 'mean', 'min', 'index_0', 'index_1', 'index_2', 'index_3', 'index_4',
                                    'index_5', 'index_6', 'index_7']
pmi_aggregation = ['max', 'mean', 'min', 'index_0', 'index_1', 'index_2', 'index_3', 'index_4', 'index_5', 'index_6',
                   'index_7', 'index_8', 'index_9', 'index_10', 'index_11']
ppl_pmi_aggregation = ['max', 'mean', 'min', 'index_0', 'index_1']
export_prefix = 'main1'

for _model, _max_length, _batch in models:
    for scoring_method in methods:
        if scoring_method in ['pmi', 'ppl_tail_masked', 'ppl_head_masked', 'ppl_add_masked'] and 'gpt' in _model:
            continue
        scorer = alm.RelationScorer(model=_model, max_length=_max_length)
        for _data in data:
            for _temp in all_templates:
                shared = dict(
                    scoring_method=scoring_method,
                    data=_data,
                    template_type=_temp,
                    batch_size=_batch,
                    export_prefix=export_prefix,
                    no_inference=True,
                    positive_permutation_aggregation=positive_permutation_aggregation,
                )
                if scoring_method == 'ppl_pmi':
                    shared['ppl_pmi_aggregation'] = ppl_pmi_aggregation
                if scoring_method == 'pmi':
                    for i in pmi_aggregation:
                        scorer.analogy_test(pmi_aggregation=i, **shared)
                        scorer.release_cache()
                else:
                    scorer.analogy_test(**shared)
                scorer.release_cache()


alm.export_report(export_prefix=export_prefix)
