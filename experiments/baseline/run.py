import alm

all_templates = [['is-to-what'], ['is-to-as'], ['rel-same'], ['what-is-to'], ['she-to-as'], ['as-what-same']]
methods = ['ppl_tail_masked', 'ppl', 'embedding_similarity', 'ppl_pmi', 'pmi']

data = ['./data/sat_package_v3.jsonl', './data/u2_raw.jsonl', './data/u4_raw.jsonl']

export_dir = './experiments/baseline/results'


def main(lm):
    for _model, _max_length, _batch in lm:
        for scoring_method in methods:
            if scoring_method in ['pmi'] and 'gpt' in _model:
                continue
            scorer = alm.RelationScorer(model=_model, max_length=_max_length)
            for _data in data:
                for _temp in all_templates:
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


if __name__ == '__main__':
    main(lm=[('roberta-large', 32, 512), ('gpt2-large', 32, 256), ('gpt2-xl', 32, 128), ('bert-large-cased', 64, 512)])
