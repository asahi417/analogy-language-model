# Grid Search (high-level search) + Test Hypothesis Only
- Data: `sat`, `u2`, `u4`, `google`, `bats`

## Target Parameters
- Language Model: `gpt2-large`, `gpt2-xl`, `bert-large-cased`, `roberta-large`
- Scoring function: `ppl`, `ppl_pmi`, `ppl_head_masked`, `ppl_tail_masked`, `ppl_add_masked`, `embeddin_similarity`, `pmi`
- Template: `is-to-what`, `is-to-as`, `rel-same`, `what-is-to`, `she-to-as`, `as-what-same`

## Script
- Get logit
```shell script
python ./experiments/main1/get_logit.py
```

- Compute score on cached logit
```shell script
python ./experiments/main1/grid_search.py
```

- Get logit & compute score on cached logit with config of best validation accuracy (Hypothesis only)
```shell script
python ./experiments/main1/run_test_hyp_only.py
```

- Get test accuracy with default configuration on every dataset and model pair
```shell script
python ./experiments/main1/run_test_default.py
```

## Output
- high-level validation accuracy: `./experiments_results/summary/main1.valid.csv`
- test accuracy (hypothesis only methods): `./experiments_results/summary/main1.test.csv`
- test accuracy (default): `./experiments_results/summary/main1.default.test.csv`