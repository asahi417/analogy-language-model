# Grid Search: Low-level search

## Target Parameters
- Language Model: `gpt2-xl`, `bert-large-cased`, `roberta-large`
- Scoring function: `ppl_pmi`
- Template: `is-to-what`, `is-to-as`, `rel-same`, `what-is-to`, `she-to-as`, `as-what-same`

## Script
- Get logit
```shell script
python ./experiments/main2/get_logit.py
```

- Compute score on cached logit
```shell script
python ./experiments/main2/grid_search.py
```

- Get logit & compute score on cached logit with config of best validation accuracy
```shell script
python ./experiments/main2/run_test_set.py
```

- Export prediction
```shell script
python ./experiments/main2/export_test_prediction.py
```

## Output
- validation accuracy: `./experiments_results/summary/main2.valid.csv`
- test accuracy: `./experiments_results/summary/main2.test.csv`
- prediction file: `./experiments_results/summary/main2.test.prediction.{data}.csv`