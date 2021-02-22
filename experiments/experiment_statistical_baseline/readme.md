# Statistical Baseline
- Random
- word2vec
- fasttext
- Glove

## Script
- Get accuracy
```shell script
python ./experiments/statistical_baseline/run.py
```

- Export a list of relation (for PMI computation) 
```shell script
python ./experiments/statistical_baseline/export_relation_list.py
```

## Output
- test accuracy: `./experiments_results/summary/statistics.test.csv`
- number of OOV: `./experiments_results/summary/statistics.test.oov.csv`