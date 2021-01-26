# Pretrained Language Model based Analogy Probing
Analogy probing via pretrained language models.


## Get started
```shell script
pip install git+https://github.com/asahi417/alm
```

## Data
We compile five different analogy task in a same format.
- BATS: [data](./data/bats), [script](./helper/compile_bats_data.py)
- Google: [data](./data/google), [script](./helper/compile_google_data.py)
- SAT: [data](./data/sat)
- Unit2: [data](./data/u2)
- Unit4: [data](./data/u4)
Each data has a validation set `valid.jsonl` and a test set `test.jsonl` as a jsonline format, where each line has 
- `stem`: a query relation pair
- `choice`: a list of candidate relation pairs
- `answer`: an index of the correct pair in candidates
- `prefix`: meta information such as level and source 