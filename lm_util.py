""" pre-trained LM for sentence evaluation """
import os
import logging
import math
from typing import List
from logging.config import dictConfig
from itertools import chain

import torch
import transformers
from tqdm import tqdm

dictConfig({
    "version": 1,
    "formatters": {'f': {'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'}},
    "handlers": {'h': {'class': 'logging.StreamHandler', 'formatter': 'f', 'level': logging.DEBUG}},
    "root": {'handlers': ['h'], 'level': logging.DEBUG}})
LOGGER = logging.getLogger()
CACHE_DIR = os.getenv("CACHE_DIR", './cache')
NUM_WORKER = os.getenv("NUM_WORKER", 1)
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning


__all__ = 'TransformersLM'


class Dataset(torch.utils.data.Dataset):
    """ `torch.utils.data.Dataset` """
    float_tensors = ['attention_mask']

    def __init__(self, data: List):
        self.data = data  # a list of dictionaries

    def __len__(self):
        return len(self.data)

    def to_tensor(self, name, data):
        if name in self.float_tensors:
            return torch.tensor(data, dtype=torch.float32)
        return torch.tensor(data, dtype=torch.long)

    def __getitem__(self, idx):
        return {k: self.to_tensor(k, v) for k, v in self.data[idx].items()}


class TransformersLM:
    """ transformers language model based sentence-mining """

    def __init__(self, model: str, max_length: int = None):
        """ transformers language model based sentence-mining

        :param model: a model name corresponding to a model card in `transformers`
        :param max_length: a model max length if specified, else use model_max_length
        """
        LOGGER.info('*** setting up a language model ***')
        # model setup
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model, cache_dir=CACHE_DIR)
        self.config = transformers.AutoConfig.from_pretrained(model, cache_dir=CACHE_DIR)
        self.model = transformers.AutoModelForMaskedLM.from_pretrained(model, config=self.config, cache_dir=CACHE_DIR)
        self.model.eval()
        if max_length:
            assert self.tokenizer.model_max_length >= max_length
            self.max_length = max_length
        else:
            self.max_length = self.tokenizer.model_max_length
        # gpu
        self.n_gpu = torch.cuda.device_count()
        assert self.n_gpu <= 1
        self.device = 'cuda' if self.n_gpu > 0 else 'cpu'
        self.model.to(self.device)
        LOGGER.info('running on %i GPU' % self.n_gpu)
        # sentence prefix tokens
        tokens = self.tokenizer.tokenize('get tokenizer specific prefix')
        tokens_encode = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode('get tokenizer specific prefix'))
        self.sp_token_prefix = tokens_encode[:tokens_encode.index(tokens[0])]
        self.sp_token_suffix = tokens_encode[tokens_encode.index(tokens[-1]) + 1:]

    def batch_encode_plus_mask(self,
                               texts: List,
                               target_tokens: List,
                               batch_size: int = 2):
        """ to get batch data_loader with `self.encode_plus_masked` function

        :param texts: a list of texts
        :param target_tokens: a list of string tokens to be masked
        :param batch_size:
        :return: `torch.utils.data.DataLoader` class
        """
        assert len(texts) == len(target_tokens), "size mismatch: {} vs {}".format(len(texts), len(target_tokens))
        data = [self.encode_plus_mask(text, token_to_mask) for text, token_to_mask in zip(texts, target_tokens)]
        data_loader = torch.utils.data.DataLoader(
            Dataset(data), num_workers=NUM_WORKER, batch_size=batch_size, shuffle=False, drop_last=False)
        return data_loader

    def encode_plus_mask(self,
                         text: str,
                         token_to_mask: str,
                         padding: bool = True):
        """ to get an output from `encode_plus` with a masked token specified by a string
        Note: it can only take single token, and a phrase over multiple tokens will raise error

        :param str text: a text to encode
        :param str token_to_mask: a target token to be masked
        :param bool padding: a flag to encode output with/without paddings
        :return encode
        """
        assert len(text.replace(' ', '')) != 0, 'found an empty text'

        token_list = self.tokenizer.tokenize(text)
        assert len(token_list) <= self.max_length,\
            "a token size exceeds the max_length: {} > {}".format(len(token_list), self.max_length)

        # get first token in the list which is exact `mask_string`
        mask_positions = [n for n, t in enumerate(token_list) if token_to_mask in t]
        assert len(mask_positions) > 0,\
            'the target token `{}` is not matched any tokens in a given text`{}`'.format(token_to_mask, token_list)
        mask_position = mask_positions[0]

        # mask the token and keep the mask position, masked token, masked token id
        # * note that `the<mask> is ~` == `the <mask> is ~` in the tokenizer module
        mask_token_id = self.tokenizer.convert_tokens_to_ids(token_list[mask_position])
        token_list[mask_position] = self.tokenizer.mask_token
        mask_position += len(self.sp_token_prefix)  # shift for prefix

        # encode sentence into model input format as a batch with single data
        encode = self.tokenizer.encode_plus(
            token_list, max_length=self.max_length, padding='max_length' if padding else False, truncation=padding)
        encode['mask_position'] = mask_position
        encode['mask_token_id'] = mask_token_id
        return encode

    def get_log_likelihood(self,
                           texts: (List, str),
                           target_tokens: (List, str),
                           batch_size: int = 2,
                           top_k_predict: int = 10):
        """ get log likelihood of a masked token within a sentence

        :param texts:
        :param target_tokens:
        :param batch_size:
        :param top_k_predict:
        :return: log_likelihood, (topk_prediction_values, topk_prediction_indices)
            log_likelihood, a list of log likelihood, (len(texts))
            topk_prediction_indices, top k tokens predicted for the masked position, (len(texts)), top_k)
            topk_prediction_values, probability along with the prediction, (len(texts)), top_k)
        """

        assert type(texts) == type(target_tokens), '`texts` and `target_tokens` should be same type'

        if type(texts) is list and type(target_tokens) is list:
            data_loader = self.batch_encode_plus_mask(texts=texts, target_tokens=target_tokens, batch_size=batch_size)
        else:
            encode = self.encode_plus_mask(text=texts, token_to_mask=target_tokens, padding=False)
            data_loader = [{k: torch.tensor([v]) for k, v in encode.items()}]

        log_likelihood, topk_prediction_indices, topk_prediction_values \
            = self.__prediction_with_data_loader(data_loader, top_k_predict=top_k_predict)
        topk_prediction_indices = [[
            self.tokenizer.decode(t) for t in topk]
            for topk in topk_prediction_indices]

        return log_likelihood, (topk_prediction_values, topk_prediction_indices)

    def batch_encode_plus_token_wise_mask(self, texts: List, batch_size: int = 2):
        """ to get batch data_loader with `self.encode_plus_token_wise_mask` function

        :param texts: a list of texts
        :param batch_size:
        :return: `torch.utils.data.DataLoader` class, partition (partition for each text)
        """
        data = [self.encode_plus_token_wise_mask(text, padding=True) for text in texts]
        length = [len(i) for i in data]
        partition = [[sum(length[:i]), sum(length[:i+1])] for i in range(len(length))]
        flatten_data = list(chain(*data))
        data_loader = torch.utils.data.DataLoader(
            Dataset(flatten_data), num_workers=NUM_WORKER, batch_size=batch_size, shuffle=False, drop_last=False)
        return data_loader, partition

    def encode_plus_token_wise_mask(self, text: str, padding: bool = True):
        """ to get a list of outputs from `encode_plus`, where each corresponds to one with a text with 
        a mask at i ~ [0, n] (n: a size of tokens in the given text) 

        :param str text: a text to encode
        :param bool padding: a flag to encode output with/without paddings
        :return encodes_list
        """
        assert len(text.replace(' ', '')) != 0, 'found an empty text'
        token_list = self.tokenizer.tokenize(text)
        assert len(token_list) <= self.max_length,\
            "a token size exceeds the max_length: {} > {}".format(len(token_list), self.max_length)

        # get encoded feature for each token masked
        def encode_with_single_mask_id(mask_position: int):
            _token_list = token_list.copy()  # can not be encode outputs because of prefix
            mask_token_id = self.tokenizer.convert_tokens_to_ids(_token_list[mask_position])
            _token_list[mask_position] = self.tokenizer.mask_token
            encode = self.tokenizer.encode_plus(
                _token_list, max_length=self.max_length, padding='max_length' if padding else False, truncation=padding)
            encode['mask_position'] = mask_position + len(self.sp_token_prefix)
            encode['mask_token_id'] = mask_token_id
            return encode

        encodes_list = [encode_with_single_mask_id(i) for i in range(len(token_list))]
        return encodes_list

    def get_pseudo_perplexity(self, texts: (List, str), batch_size: int = 2):
        """ to compute a pseudo perplexity (mask each token and use log likelihood for each prediction for the mask)

        :param texts:
        :param batch_size:
        :return:
        """
        if type(texts) is list:
            data_loader, partition = self.batch_encode_plus_token_wise_mask(texts, batch_size=batch_size)
        else:
            encode_list = self.encode_plus_token_wise_mask(texts)
            data_loader = [{k: torch.tensor([v]) for k, v in encode.items()} for encode in encode_list]
            partition = [[0, len(data_loader)]]

        list_ppl = []
        loglikelihood, _, _ = self.__prediction_with_data_loader(data_loader)
        for start, end in partition:
            sentence_loglikeli = loglikelihood[start:end]
            list_ppl += [math.exp(- sum(sentence_loglikeli) / len(sentence_loglikeli))]

        return list_ppl

    def __prediction_with_data_loader(self, data_loader, top_k_predict: int = None):
        """ to run a prediction on a masked token with MLM

        :param data_loader:
        :param top_k_predict:
        :return:
        """
        topk_prediction_indices = []
        topk_prediction_values = []
        log_likelihood = []
        with torch.no_grad():
            for encode in tqdm(data_loader):
                mask_position = encode.pop('mask_position').cpu().detach().int().tolist()
                mask_token_id = encode.pop('mask_token_id').cpu().detach().int().tolist()

                # get probability/prediction for batch features
                encode = {k: v.to(self.device) for k, v in encode.items()}
                logit = self.model(**encode)[0]
                prob = torch.softmax(logit, -1)

                # compute likelihood of masked positions given the masked tokens
                log_likelihood += [
                    math.log(float(prob[n][m_p][m_i].cpu())) for n, (m_p, m_i)
                    in enumerate(zip(mask_position, mask_token_id))]

                # top-k prediction
                if top_k_predict:
                    top_k = [
                        [i.cpu().tolist() for i in prob[n][m_p].topk(top_k_predict)]
                        for n, m_p in enumerate(mask_position)]
                    topk_prediction_values += list(list(zip(*top_k))[0])
                    topk_prediction_indices += list(list(zip(*top_k))[1])

        return log_likelihood, topk_prediction_indices, topk_prediction_values
