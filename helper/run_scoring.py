import alm
scorer = alm.RelationScorer(model='roberta-large')

path_to_data = './data/sat_package_v3.jsonl'
template_types = ['is-to-what']
batch_size = 16


for permutation_positive in [True, False]:
    if permutation_positive:
        for aggregation_positive in ['max', 'mean', 'min']:
            for permutation_negative in [True, False]:
                if permutation_negative:
                    for aggregation_negative in ['max', 'mean', 'min']:
                        scorer.analogy_test(
                            path_to_data=path_to_data, template_types=template_types, batch_size=batch_size,
                            permutation_positive=permutation_positive, aggregation_positive=aggregation_positive,
                            permutation_negative=permutation_negative, aggregation_negative=aggregation_negative)
                else:
                    aggregation_negative = 'none'
                    scorer.analogy_test(
                        path_to_data=path_to_data, template_types=template_types, batch_size=batch_size,
                        permutation_positive=permutation_positive, aggregation_positive=aggregation_positive,
                        permutation_negative=permutation_negative, aggregation_negative=aggregation_negative)

    else:
        aggregation_positive = 'none'
        for permutation_negative in [True, False]:
            if permutation_negative:
                for aggregation_negative in ['max', 'mean', 'min']:
                    scorer.analogy_test(
                        path_to_data=path_to_data, template_types=template_types, batch_size=batch_size,
                        permutation_positive=permutation_positive, aggregation_positive=aggregation_positive,
                        permutation_negative=permutation_negative, aggregation_negative=aggregation_negative)
            else:
                aggregation_negative = 'none'
                scorer.analogy_test(
                    path_to_data=path_to_data, template_types=template_types, batch_size=batch_size,
                    permutation_positive=permutation_positive, aggregation_positive=aggregation_positive,
                    permutation_negative=permutation_negative, aggregation_negative=aggregation_negative)
