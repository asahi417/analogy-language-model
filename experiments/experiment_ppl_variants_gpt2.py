import json
import alm
import pandas as pd

alm.util.fix_seed(1234)
scorer = alm.RelationScorer(model='gpt2-xl', max_length=256)
scorer.analogy_test(
    scoring_method='ppl',
    data='sat',
    template_type='is-to-as',
    batch_size=32,
    negative_permutation=False,
    test=True
)
scorer.release_cache()
alm.export_report(export_prefix='gpt2_result', test=True)
