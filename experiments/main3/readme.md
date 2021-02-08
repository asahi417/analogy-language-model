# Grid Search: on test set

## Target Parameters
- Language Model: `gpt2-xl`, `roberta-large`
- Scoring function: `ppl_pmi`
- Template: `is-to-what`, `is-to-as`, `rel-same`, `what-is-to`, `she-to-as`, `as-what-same`

## Script
- Get logit
```shell script
python ./experiments/tune_on_test/get_logit.py
```

- Compute score on cached logit
```shell script
python ./experiments/tune_on_test/grid_search.py
```


## Output
- test accuracy: `./experiments_results/summary/tune_on_test.test.csv`