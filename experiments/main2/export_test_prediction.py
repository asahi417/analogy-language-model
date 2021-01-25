""" Export prediction file of the best configuration in test accuracy """
import logging
import json
from pprint import pprint
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
import alm

data = ['sat', 'u2', 'u4', 'google', 'bats']
export_prefix = 'main2'
df = alm.get_report(export_prefix=export_prefix, test=True)

for i in data:
    tmp_df = df[df.data == i]
    val_accuracy = tmp_df.sort_values(by='accuracy', ascending=False).head(1)['accuracy'].values[0]
    logging.info("RUN TEST:\n - data: {} \n - validation accuracy: {} ".format(i, val_accuracy))
    best_configs = tmp_df[tmp_df['accuracy'] == val_accuracy]
    logging.info("find {} configs with same accuracy".format(len(best_configs)))
    accuracy_list = [json.loads(tmp_df.to_json())['accuracy'] for n, tmp_df in best_configs.iterrows()]
    logging.info("min accuracy: {}".format(min(accuracy_list)))
    logging.info("max accuracy: {}".format(max(accuracy_list)))
    config = json.loads(best_configs.iloc[0].to_json())
    logging.info("use the first one: {} accuracy".format(config.pop('accuracy')))
    scorer = alm.RelationScorer(model=config.pop('model'), max_length=config.pop('max_length'))
    scorer.analogy_test(test=True,
                        export_prediction=True,
                        no_inference=True,
                        export_prefix=export_prefix,
                        **config)
    scorer.release_cache()
