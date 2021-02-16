import alm

all_templates = ['is-to-what', 'is-to-as', 'rel-same', 'what-is-to', 'she-to-as', 'as-what-same']
data = ['sat']
# models = [('roberta-large', 32, 512), ('bert-large-cased', 32, 1024)]
models = [('bert-large-cased', 32, 1024)]
positive_permutation_aggregation = ['max', 'mean', 'min', 'index_0', 'index_1', 'index_2', 'index_3', 'index_4',
                                    'index_5', 'index_6', 'index_7']
negative_permutation_aggregation = ['max', 'mean', 'min', 'index_0', 'index_1', 'index_2', 'index_3', 'index_4',
                                    'index_5', 'index_6', 'index_7', 'index_8', 'index_9', 'index_10', 'index_11']
negative_permutation_weight = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
ppl_hyp_eta_head = [-0.4, -0.2, 0, 0.2, 0.4]
ppl_hyp_eta_tail = [-0.4, -0.2, 0, 0.2, 0.4]
export_prefix = 'main4.ppl_hyp'

for _model, _max_length, _batch in models:
    scorer = alm.RelationScorer(model=_model, max_length=_max_length)
    for _data in data:
        for _temp in all_templates:
            for test in [False, True]:
                scorer.analogy_test(
                    scoring_method='ppl_hyp',
                    data=_data,
                    template_type=_temp,
                    batch_size=_batch,
                    export_prefix=export_prefix,
                    ppl_hyp_eta_head=ppl_hyp_eta_head,
                    ppl_hyp_eta_tail=ppl_hyp_eta_tail,
                    negative_permutation=True,
                    positive_permutation_aggregation=positive_permutation_aggregation,
                    negative_permutation_aggregation=negative_permutation_aggregation,
                    negative_permutation_weight=negative_permutation_weight,
                    test=test)
                scorer.release_cache()

alm.export_report(export_prefix=export_prefix, test=True)
