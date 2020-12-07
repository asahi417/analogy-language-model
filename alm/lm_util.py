""" pre-trained LM for sentence evaluation """
import os
import re
import logging
import math
from typing import List
from itertools import chain
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

import transformers
import torch
from torch import nn
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning message
PAD_TOKEN_LABEL_ID = nn.CrossEntropyLoss().ignore_index


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

    def __init__(self,
                 model: str,
                 max_length: int = None,
                 cache_dir: str = './cache',
                 num_worker: int = 1,
                 embedding_mode: bool = False):
        """ transformers language model based sentence-mining

        :param model: a model name corresponding to a model card in `transformers`
        :param max_length: a model max length if specified, else use model_max_length
        """
        logging.info('*** setting up a language model ***')
        self.num_worker = num_worker
        # model setup
        self.is_causal = 'gpt' in model  # TODO: fix to be more comprehensive method
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)
        self.config = transformers.AutoConfig.from_pretrained(model, cache_dir=cache_dir)
        params = dict(config=self.config, cache_dir=cache_dir)
        self.embedding_mode = embedding_mode
        if embedding_mode:
            # a mode to use embedding instead of prediction logit
            self.model = transformers.AutoModel.from_pretrained(model, **params)
        elif self.is_causal:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(model, **params)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.model = transformers.AutoModelForMaskedLM.from_pretrained(model, **params)
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
        logging.info('running on %i GPU' % self.n_gpu)
        # sentence prefix tokens
        tokens = self.tokenizer.tokenize('get tokenizer specific prefix')
        tokens_encode = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode('get tokenizer specific prefix'))
        self.sp_token_prefix = tokens_encode[:tokens_encode.index(tokens[0])]
        self.sp_token_suffix = tokens_encode[tokens_encode.index(tokens[-1]) + 1:]

    def find_position(self, str_to_mask, text, token: List = None):
        """ find masking position in token space given a string target

        :param str_to_mask: a string to be masked
        :param text: source text
        :param token: (optional) tokenized text
        :return: [start_position, end_position] in token space
        """
        mask_trg = re.findall(r'\s*\b{}\b'.format(str_to_mask), text)
        if len(mask_trg) == 0:
            raise ValueError('mask target {} not found in source {}'.format(str_to_mask, text))
        if len(mask_trg) > 1:
            raise ValueError('mask target {} found multiple times in source {}'.format(str_to_mask, text))
        mask_trg = mask_trg[0]
        token_before = self.tokenizer.tokenize(text[:text.index(mask_trg)])
        if token is None:
            token = self.tokenizer.tokenize(text)
        assert token[:len(token_before)] == token_before, 'wrong token\n `{}` vs `{}`'.format(token_before, token)
        i = len(token_before)
        while i < len(token):
            i += 1
            decode = self.tokenizer.convert_tokens_to_string(token[:i])
            if str_to_mask in decode:
                break
        return [len(token_before), i]

    def input_ids_to_labels(self, input_ids, label_position: List = None):
        """ replace pad_token_id by token which is ignored when loss computation """
        if label_position is None:
            return list(map(lambda x: PAD_TOKEN_LABEL_ID if x == self.tokenizer.pad_token_id else x, input_ids))
        return list(map(
            lambda x: PAD_TOKEN_LABEL_ID if x[1] == self.tokenizer.pad_token_id or x[0] not in label_position else x[1],
            enumerate(input_ids)
        ))

    def encode_plus_mask(self,
                         text: str,
                         token_to_embed: List = None,
                         token_to_mask: List = None,
                         token_to_label: List = None):
        """ to get an output from `encode_plus` with a masked token specified by a string
        Note: it can only take single token, and a phrase over multiple tokens will raise error

        :param str text: a text to encode
        :param token_to_mask: a target token to be masked
        :param token_to_embed:
        :param token_to_label:
        :return encode
        """
        assert type(text) is str
        assert len(text.replace(' ', '')) != 0, 'found an empty text'
        param = {'max_length': self.max_length, 'truncation': True, 'padding': 'max_length'}

        if self.is_causal or self.embedding_mode:
            assert token_to_mask is None, 'token_to_mask is not for either causalLM or embedding mode'
            assert token_to_label is None, 'token_to_label is not for either causalLM or embedding mode'
            encode = self.tokenizer.encode_plus(text, **param)
            if self.embedding_mode:
                token_list = self.tokenizer.tokenize(text)
                if token_to_embed is not None:
                    positions = [self.find_position(s, text, token_list) for s in token_to_embed]
                    pos = [[len(self.sp_token_prefix) + s, len(self.sp_token_prefix) + e] for s, e in positions]
                    assert len(pos) <= 5, 'token_to_embed is allowed upto 5'
                    encode['position_to_embed'] = pos + [[0, 0]] * (5 - len(pos))
                else:
                    encode['position_to_embed'] = \
                        [[len(self.sp_token_prefix), len(self.sp_token_prefix) + len(token_list)]] + [[0, 0]] * 4
            else:
                encode['labels'] = self.input_ids_to_labels(encode['input_ids'])
            return [encode]
        else:
            token_list = self.tokenizer.tokenize(text)

            if token_to_mask is not None:  # mask specified token
                mask_positions = [self.find_position(s, text, token_list) for s in token_to_mask]
                # mask the token and keep the mask position, masked token, masked token id
                # * note that `the<mask> is ~` == `the <mask> is ~` in the tokenizer module
                if token_to_label is None:
                    token_to_label = token_to_mask
                else:
                    assert all(i in token_to_mask for i in token_to_label), 'invalid token_to_label'
                label_position = []
                for (s, e), token_m in zip(mask_positions, token_to_mask):
                    if token_m in token_to_label:
                        label_position += list(range(s, e))
                    token_list[s:e] = [self.tokenizer.mask_token] * (e - s)
                # encode sentence into model input format as a batch with single data
                encode = self.tokenizer.encode_plus(token_list, **param)
                encode['labels'] = self.input_ids_to_labels(encode['input_ids'], label_position=label_position)
                return [encode]

            else:  # token-wise mask
                def encode_with_single_mask_id(mask_position: int):
                    _token_list = token_list.copy()  # can not be encode outputs because of prefix
                    _token_list[mask_position] = self.tokenizer.mask_token
                    _encode = self.tokenizer.encode_plus(_token_list, **param)
                    _encode['labels'] = self.input_ids_to_labels(
                        _encode['input_ids'], label_position=[mask_position + len(self.sp_token_prefix)])
                    return _encode

                return [encode_with_single_mask_id(i) for i in range(len(token_list))]

    def batch_encode_plus_mask(self,
                               batch_text: List,
                               batch_token_to_mask: List = None,
                               batch_token_to_embed: List = None,
                               batch_size: int = None):
        """ to get batch data_loader with `self.encode_plus_masked` function

        :param batch_text: a list of texts
        :param batch_token_to_mask: a list of lists for masking
        :param batch_token_to_embed:
        :param batch_size:
        :return: `torch.utils.data.DataLoader` class, partition
        """
        if self.num_worker == 1:
            os.environ["OMP_NUM_THREADS"] = "1"  # to turn off warning message

        batch_size = len(batch_text) if batch_size is None else batch_size
        batch_token_to_mask = [None] * len(batch_text) if batch_token_to_mask is None else batch_token_to_mask
        batch_token_to_embed = [None] * len(batch_text) if batch_token_to_embed is None else batch_token_to_embed
        assert len(batch_text) == len(batch_token_to_mask),\
            "size mismatch: {} vs {}".format(len(batch_text), len(batch_token_to_mask))

        data = list(map(
            lambda x: self.encode_plus_mask(text=x[0], token_to_mask=x[1], token_to_embed=x[2]),
            zip(batch_text, batch_token_to_mask, batch_token_to_embed)
        ))
        length = list(map(lambda x: len(x), data))
        partition = list(map(lambda x: [sum(length[:x]), sum(length[:x + 1])], range(len(length))))
        param = dict(num_workers=self.num_worker, batch_size=batch_size, shuffle=False, drop_last=False)
        data_loader = torch.utils.data.DataLoader(Dataset(list(chain(*data))), **param)
        return data_loader, partition

    def get_perplexity(self, texts: List, batch_size: int = None):
        """ to compute perplexity

        :param texts:
        :param batch_size:
        :return: a list of ppl
        """
        assert type(texts) is list, 'type error'
        assert not self.embedding_mode
        data_loader, partition = self.batch_encode_plus_mask(texts, batch_size=batch_size)
        nll = self.__get_nll(data_loader)
        # for pseudo likelihood aggregation
        return list(map(lambda x: math.exp(sum(nll[x[0]:x[1]]) / (x[1] - x[0])), partition))

    def get_nll(self, texts: List, tokens_to_mask: List, batch_size: int = None):
        """ to compute negative log likelihood of a masked token

        :param texts:
        :param tokens_to_mask:
        :param batch_size:
        :return: negative_log_likelihood
        """
        assert type(texts) is list and type(tokens_to_mask) is list, 'type error'
        assert not self.embedding_mode or self.is_causal
        data_loader, _ = self.batch_encode_plus_mask(texts, batch_token_to_mask=tokens_to_mask, batch_size=batch_size)
        return self.__get_nll(data_loader)

    def __get_nll(self, data_loader):
        """ get negative log likelihood with a dataloader """
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        nll = []
        with torch.no_grad():
            for encode in tqdm(data_loader):
                encode = {k: v.to(self.device) for k, v in encode.items()}
                labels = encode.pop('labels')
                output = self.model(**encode, return_dict=True)
                prediction_scores = output['logits']
                loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
                loss = torch.sum(loss.view(len(prediction_scores), -1), -1)
                # print(list(map(
                #     lambda x: sum(map(lambda y: y != PAD_TOKEN_LABEL_ID, x[1])),
                #     zip(loss.cpu().tolist(), labels.cpu().tolist())
                # )))
                nll += list(map(
                    lambda x: x[0]/sum(map(lambda y: y != PAD_TOKEN_LABEL_ID, x[1])),
                    zip(loss.cpu().tolist(), labels.cpu().tolist())
                ))
        print(nll)

        return nll

    def get_embedding(self, texts: List, tokens_to_embed: List, batch_size: int = None):
        """ get embedding

        :param texts:
        :param tokens_to_embed:
        :param batch_size:
        :return: embeddings (len(texts), token num, dim)
        """
        assert self.embedding_mode
        data_loader, _ = self.batch_encode_plus_mask(
            texts, batch_size=batch_size, batch_token_to_embed=tokens_to_embed)
        embeddings = []
        with torch.no_grad():
            for encode in tqdm(data_loader):
                position_to_embed = encode.pop('position_to_embed').cpu().tolist()
                encode = {k: v.to(self.device) for k, v in encode.items()}
                output = self.model(**encode, return_dict=True)
                last_hidden_state = output['last_hidden_state']  # batch, length, dim
                # labels  # batch, target_tokens, (start, end)
                embeddings += [
                    [torch.mean(last_hidden_state[n][s:e], 0).cpu().tolist() for s, e in positions if s != 0 and e != 0]
                    for n, positions in enumerate(position_to_embed)
                ]  # batch, target_tokens, dim

        return embeddings
