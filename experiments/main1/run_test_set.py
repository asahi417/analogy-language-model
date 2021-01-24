import logging
import json
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
import alm

data = ['sat', 'u2', 'u4', 'google', 'bats']
export_prefix = 'main1'
df = alm.get_report(export_prefix=export_prefix)

for i in data:
    tmp_df = df[df.data == i]
    tmp_df = tmp_df.sort_values(by='accuracy', ascending=False).iloc[0]
    config = json.loads(tmp_df.to_json())
    logging.info("RUN TEST:\n - data: {} \n - validation accuracy: {} ".format(i, config.pop('accuracy')))
    scorer = alm.RelationScorer(model=config.pop('model'), max_length=config.pop('max_length'))
    scorer.analogy_test(test=True, batch_size=512, **config)
    scorer.release_cache()

alm.export_report(export_prefix=export_prefix, test=True)
