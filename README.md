# BERT is to NLP what AlexNet is to CV
This is the official implementation of ***BERT is to NLP what AlexNet is to CV: Can Pre-Trained Language Models Identify Analogies?*** (the camera-ready version of the paper is [here](https://aclanthology.org/2021.acl-long.280/))
which has been accepted by the **[ACL 2021 main conference](https://2021.aclweb.org/)**. We evaluate pretrained language models (LM) on five analogy tests that follow SAT-style format as below.
```
QUERY word:language
OPTION
  (1) paint:portrait
  (2) poetry:rhythm 
  (3) note:music <-- the answer!
  (4) tale:story
  (5) week:year 
```
We devise a new class of scoring functions, referred to as *analogical proportion (AP)* scores, to solve word analogies in an unsurpervised fashion and investigate the relational knowledge that LM learnt through pretraining.
<p align="center">
  <img src="asset/overview.png" width="500">
</p>   

Please see our paper for more information and discussion.

## Get started
```shell
git clone https://github.com/asahi417/analogy-language-model
cd analogy-language-model
pip install -e .
```

## Run Experiments
The following scripts reproduce our results in the paper.
```shell
# get result for our main AP score
python experiments/experiment_ppl_variants.py 
# get result for word embedding baseline
python experiments/experiment_word_embedding.py 
# get result for other scoring function such as vector difference, etc
python experiments/experiment_scoring_comparison.py 
```
Here's the result summary that can be attained by running those scripts.
- [experimental results](https://github.com/asahi417/alm/releases/download/0.0.0/experiments_results.tar.gz)

## Dataset
The datasets used in our experiments can be downloaded from the following link:
- [Analogy Datasets](https://github.com/asahi417/AnalogyTools/releases/download/0.0.0/analogy_test_dataset.zip)

Please see [the Analogy Tool](https://github.com/asahi417/AnalogyTools) for more information about the dataset and baselines.

## Citation
Please cite our [reference paper](https://aclanthology.org/2021.acl-long.280/) if you use our data or code:
```
@inproceedings{ushio-etal-2021-bert,
    title = "{BERT} is to {NLP} what {A}lex{N}et is to {CV}: Can Pre-Trained Language Models Identify Analogies?",
    author = "Ushio, Asahi  and
      Espinosa Anke, Luis  and
      Schockaert, Steven  and
      Camacho-Collados, Jose",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.280",
    doi = "10.18653/v1/2021.acl-long.280",
    pages = "3609--3624",
    abstract = "Analogies play a central role in human commonsense reasoning. The ability to recognize analogies such as {``}eye is to seeing what ear is to hearing{''}, sometimes referred to as analogical proportions, shape how we structure knowledge and understand language. Surprisingly, however, the task of identifying such analogies has not yet received much attention in the language model era. In this paper, we analyze the capabilities of transformer-based language models on this unsupervised task, using benchmarks obtained from educational settings, as well as more commonly used datasets. We find that off-the-shelf language models can identify analogies to a certain extent, but struggle with abstract and complex relations, and results are highly sensitive to model architecture and hyperparameters. Overall the best results were obtained with GPT-2 and RoBERTa, while configurations using BERT were not able to outperform word embedding models. Our results raise important questions for future work about how, and to what extent, pre-trained language models capture knowledge about abstract semantic relations.",
}
```

Please also cite the relevant reference papers if using any of the analogy datasets.
