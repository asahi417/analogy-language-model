""" Run on test set
- LM x Dataset x [PPL-based, PMI-based]
where PPL-based is ppl_pmi_marginal_version == True
and PMI-based is ppl_pmi_marginal_version == False
"""
import logging
import json
from itertools import product
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
import alm

data = ['sat', 'u2', 'u4', 'google', 'bats']
models = [('roberta-large', 32, 512), ('gpt2-xl', 32, 128), ('bert-large-cased', 32, 1024)]
export_prefix = 'main2.ppl_pmi'
df = alm.get_report(export_prefix=export_prefix)

for i, m in product(data, models):
    _model, _len, _batch = m
    tmp_df_ = df[df.data == i]
    tmp_df_ = tmp_df_[tmp_df_.model == _model]
    for v in [True, False]:
        tmp_df = tmp_df_[tmp_df_.ppl_pmi_marginal_version == v]
        val_accuracy = tmp_df.sort_values(by='accuracy', ascending=False).head(1)['accuracy'].values[0]
        logging.info("RUN TEST:\n - data: {} \n - lm: {} \n - validation accuracy: {} ".format(i, _model, val_accuracy))
        best_configs = tmp_df[tmp_df['accuracy'] == val_accuracy]
        logging.info("find {} configs with same accuracy".format(len(best_configs)))
        for n, tmp_df in best_configs.iterrows():
            config = json.loads(tmp_df.to_json())
            val_accuracy = config.pop('accuracy')
            assert _len == config.pop('max_length')
            scorer = alm.RelationScorer(model=config.pop('model'), max_length=_len)
            scorer.analogy_test(
                test=True, export_prefix=export_prefix, batch_size=_batch, val_accuracy=val_accuracy, **config)
            scorer.release_cache()

alm.export_report(export_prefix=export_prefix, test=True)