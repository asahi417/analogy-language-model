import logging
import json
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
import alm

data = ['sat', 'u2', 'u4', 'google', 'bats']
export_prefix = 'main2'
df = alm.get_report(export_prefix=export_prefix)

for i in data:
    tmp_df = df[df.data == i]
    val_accuracy = tmp_df.sort_values(by='accuracy', ascending=False).head(1)['accuracy'].values[0]
    logging.info("RUN TEST:\n - data: {} \n - validation accuracy: {} ".format(i, val_accuracy))
    best_configs = tmp_df[tmp_df['accuracy'] == val_accuracy]
    logging.info("find {} configs with same accuracy".format(len(best_configs)))
    for n, tmp_df in best_configs.iterrows():
        config = json.loads(tmp_df.to_json())
        config.pop('accuracy')
        scorer = alm.RelationScorer(model=config.pop('model'), max_length=config.pop('max_length'))
        scorer.analogy_test(test=True,
                            export_prefix=export_prefix,
                            batch_size=128 if 'gpt2-xl' == scorer.model_name else 512,
                            **config)
        scorer.release_cache()

alm.export_report(export_prefix=export_prefix, test=True)
