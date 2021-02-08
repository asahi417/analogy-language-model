""" Export prediction file of the best configuration in test accuracy """
import logging
import json
from itertools import product
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
import alm

export_prefix = 'main2'
models = ['roberta-large', 'gpt2-xl', 'bert-large-cased']
data = ['sat', 'u2', 'u4', 'google', 'bats']
df = alm.get_report(export_prefix=export_prefix, test=True)

for m, d in product(models, data):
    tmp_df = df[df.data == d]
    tmp_df = tmp_df[tmp_df.model == m]
    accuracy = tmp_df.sort_values(by='accuracy', ascending=False).head(1)['accuracy'].values[0]
    logging.info("RUN TEST:\n - data: {} \n - test accuracy: {} ".format(d, accuracy))
    best_configs = tmp_df[tmp_df['accuracy'] == accuracy]
    # assert len(best_configs) == 1, str(best_configs)
    config = json.loads(best_configs.iloc[0].to_json())
    logging.info("use the first one: {} accuracy".format(config.pop('accuracy')))
    scorer = alm.RelationScorer(model=config.pop('model'), max_length=config.pop('max_length'))
    scorer.analogy_test(test=True, export_prediction=True, no_inference=True, export_prefix=export_prefix, **config)
    scorer.release_cache()
