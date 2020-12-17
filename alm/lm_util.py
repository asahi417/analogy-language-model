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

from .dict_keeper import DictKeeper

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning message
PAD_TOKEN_LABEL_ID = nn.CrossEntropyLoss().ignore_index


__all__ = 'TransformersLM'


def get_partition(_list):
    length = list(map(lambda x: len(x), _list))
    return list(map(lambda x: [sum(length[:x]), sum(length[:x + 1])], range(len(length))))


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
                 num_worker: int = 1):
        """ transformers language model based sentence-mining

        :param model: a model name corresponding to a model card in `transformers`
        :param max_length: a model max length if specified, else use model_max_length
        """
        logging.info('*** setting up a language model ***')
        self.num_worker = num_worker
        if self.num_worker == 1:
            os.environ["OMP_NUM_THREADS"] = "1"  # to turn off warning message

        self.model_name = model
        self.cache_dir = cache_dir
        self.device = 'cpu'
        self.model = None
        self.is_causal = 'gpt' in self.model_name  # TODO: fix to be more comprehensive method
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)
        if self.is_causal:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.config = transformers.AutoConfig.from_pretrained(model, cache_dir=cache_dir)
        if max_length:
            assert self.tokenizer.model_max_length >= max_length
            self.max_length = max_length
        else:
            self.max_length = self.tokenizer.model_max_length

        # sentence prefix tokens
        tokens = self.tokenizer.tokenize('get tokenizer specific prefix')
        tokens_encode = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode('get tokenizer specific prefix'))
        self.sp_token_prefix = tokens_encode[:tokens_encode.index(tokens[0])]
        self.sp_token_suffix = tokens_encode[tokens_encode.index(tokens[-1]) + 1:]

    def load_model(self, lm_head: bool = True):
        """ Model setup """
        logging.info('setting up language model')
        params = dict(config=self.config, cache_dir=self.cache_dir)
        if lm_head and self.is_causal:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_name, **params)
        elif lm_head:
            self.model = transformers.AutoModelForMaskedLM.from_pretrained(self.model_name, **params)
        else:
            self.model = transformers.AutoModel.from_pretrained(self.model_name, **params)
        self.model.eval()
        # gpu
        n_gpu = torch.cuda.device_count()
        assert n_gpu <= 1
        self.device = 'cuda' if n_gpu > 0 else 'cpu'
        self.model.to(self.device)
        logging.info('running on {} GPU'.format(n_gpu))

    def find_position(self, str_to_mask, text, token: List = None):
        """ Find masking position in a token-space given a string target

        :param str_to_mask: a string to be masked
        :param text: source text
        :param token: (optional) tokenized text
        :return: [start_position, end_position] in a token-space
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
            if str_to_mask in decode or str_to_mask.lower() in decode:
                break
        return [len(token_before), i]

    def input_ids_to_labels(self, input_ids, label_position: List = None, label_id: List = None):
        """ Labels generation for loss computation

        :param input_ids: input_ids given by tokenizer.encode
        :param label_position: position to keep for label
        :param label_id: indices to use in `label_position`
        :return: labels, a list of indices for loss computation
        """
        if label_position is None and label_id is None:
            # ignore padding token
            label = list(map(lambda x: PAD_TOKEN_LABEL_ID if x == self.tokenizer.pad_token_id else x, input_ids))
        else:
            assert len(label_position) == len(label_id)
            label = [PAD_TOKEN_LABEL_ID] * len(input_ids)
            for p, i in zip(label_position, label_id):
                label[p] = i
        if self.is_causal:  # shift the label sequence for causal inference
            label = label[1:] + [PAD_TOKEN_LABEL_ID]
        return label

    def __get_nll(self, data_loader, reduce: bool = True):
        """ Negative log likelihood (NLL)

        :param data_loader: data loader
        :param reduce: to reduce NLL over sequence or not
        :return: a list of NLL
        """
        assert self.model
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        nll = []
        with torch.no_grad():
            for encode in tqdm(data_loader):
                encode = {k: v.to(self.device) for k, v in encode.items()}
                labels = encode.pop('labels')
                output = self.model(**encode, return_dict=True)
                prediction_scores = output['logits']
                loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
                loss = loss.view(len(prediction_scores), -1)

                if reduce:
                    loss = torch.sum(loss, -1)
                    nll += list(map(
                        lambda x: x[0] / sum(map(lambda y: y != PAD_TOKEN_LABEL_ID, x[1])),
                        zip(loss.cpu().tolist(), labels.cpu().tolist())
                    ))
                else:
                    nll += list(map(
                        lambda x: list(map(
                            lambda y: y[1],
                            filter(lambda z: z[0] != PAD_TOKEN_LABEL_ID, zip(x[0], x[1]))
                        )),
                        zip(labels.cpu().tolist(), loss.cpu().tolist())))

        return nll

    ########################################
    # Modules for negative PMI computation #
    ########################################
    def encode_plus_mask(self,
                         text: str,
                         token_to_mask: str,
                         token_to_mask_no_label: str = None):
        """ An output from `encode_plus` with a masked token specified by a string with a `labels` indicating
        the masking position as the masked token id otherwise `PAD_TOKEN_LABEL_ID`
        * Token with multiple sub-words includes all the possible decoding paths

        :param str text: a text to encode
        :param token_to_mask: a token to be masked
        :param token_to_mask_no_label: a token to be masked and skip in labels (masked but ignored when computing loss)
        :return `DictKeeper` object
        """
        assert type(text) is str
        assert len(text.replace(' ', '')) != 0, 'found an empty text'
        assert not self.is_causal

        token_list = self.tokenizer.tokenize(text)
        token_list_tmp = token_list.copy()
        if token_to_mask_no_label is not None:
            s, e = self.find_position(token_to_mask_no_label, text, token_list)
            token_list_tmp[s:e] = [self.tokenizer.mask_token] * (e - s)

        s, e = self.find_position(token_to_mask, text, token_list)
        all_encode = self.encode_combinations(token_list_tmp, list(range(s, e)))
        return DictKeeper(all_encode, target_key='encode')

    def encode_combinations(self, token_list, positions: List):
        """ Encode all the combination of positions

        :param token_list: a list of token
        :param positions: a list of position (index)
        :return: a nested dictionary consisting of masked encoding with all position patterns
        """

        def pop(__list, value):
            __list = __list.copy()
            __list.pop(__list.index(value))
            _out = {"index": __list, "encode": self.encode_position(token_list, __list)}
            if len(__list) > 1:
                _out['child'] = {_i: pop(__list, _i) for _i in __list}
            return _out

        assert len(positions) != 0
        out = {"index": positions, "encode": self.encode_position(token_list, positions)}
        if len(positions) != 1:
            out['child'] = {i: pop(positions, i) for i in positions}
        return out

    def encode_position(self, token_list, position: List):
        """ Encode tokens with masks at a position

        :param token_list: a list of token
        :param position: a position eg) (1, 2)
        :return: encode with labels by `token_list` with masks at `_positions`
        """
        param = {'max_length': self.max_length, 'truncation': True, 'padding': 'max_length'}
        tmp_token_list = token_list.copy()
        label_id = []
        for _p in position:
            label_id.append(self.tokenizer.convert_tokens_to_ids(tmp_token_list[_p]))
            tmp_token_list[_p] = self.tokenizer.mask_token
        _encode = self.tokenizer.encode_plus(tmp_token_list, **param)
        _encode['labels'] = self.input_ids_to_labels(
            _encode['input_ids'], label_position=position, label_id=label_id)
        return _encode

    def batch_encode_plus_mask(self,
                               batch_text: List,
                               batch_token_to_mask: List,
                               batch_token_to_mask_no_label: List = None,
                               batch_size: int = None):
        """ Batch version of `self.encode_plus_mask`

        :param batch_text: a batch `text`
        :param batch_size: batch size
        :param batch_token_to_mask: a batch of `token_to_mask`
        :param batch_token_to_mask_no_label: a batch of `token_to_mask_no_label`
        :return: (`torch.utils.data.DataLoader` class, partition)
        """

        batch_size = len(batch_text) if batch_size is None else batch_size
        if batch_token_to_mask_no_label is None:
            batch_token_to_mask_no_label = [None] * len(batch_text)

        assert len(batch_text) == len(batch_token_to_mask) == len(batch_token_to_mask_no_label)

        logging.info('creating data loader')
        data_dk = []
        # this can be parallelized, but due to deepcopy at DictKeeper, it may cause memory error in some machine
        for x in tqdm(zip(batch_text, batch_token_to_mask, batch_token_to_mask_no_label)):
            data_dk.append(self.encode_plus_mask(text=x[0], token_to_mask=x[1], token_to_mask_no_label=x[2]))
        # data_dk = list(map(
        #     lambda x: self.encode_plus_mask(text=x[0], token_to_mask=x[1], token_to_mask_no_label=x[2]),
        #     zip(batch_text, batch_token_to_mask, batch_token_to_mask_no_label)
        # ))

        data_flat = list(map(lambda x: x.flat_values, data_dk))
        partition = get_partition(data_flat)

        data_loader = torch.utils.data.DataLoader(
            Dataset(list(chain(*data_flat))),
            num_workers=self.num_worker, batch_size=batch_size, shuffle=False, drop_last=False
        )
        return data_loader, partition, data_dk

    def get_negative_pmi(self,
                         texts: List,
                         tokens_to_mask: List,
                         tokens_to_condition: List = None,
                         batch_size: int = None,
                         weight: float = None):
        """ Negative Point-wise Mutual Information (PMI)
        negative PMI(t|c) = w * sum[NLL(t)] - sum[NLL(t|c=mask])]
        - NLL(t): negative log likelihood of t
        - t: objective (sub) tokens
        - c: conditioning (sub) tokens
        - w: conditional NLL weight
        Sum over (sub) tokens are based on lowest NLL search

        :param texts: a batch of texts
        :param tokens_to_mask: a batch of objective tokens
        :param tokens_to_condition: a batch of conditioning tokens
        :param batch_size: batch size
        :param weight: conditional NLL weight
        :return:
        """
        assert type(texts) is list and type(tokens_to_mask) is list, 'type error'
        if not self.model:
            self.load_model()
        weight = 1 if weight is None else weight

        def decode_score(_nested_score, total_score: float = 0.0):
            """ Lowest nll based subword decoding """
            scores = _nested_score['score']
            if len(scores) == 1:
                assert 'child' not in _nested_score.keys()
                return total_score + scores[0]
            else:
                assert 'child' in _nested_score.keys()
                assert len(_nested_score['score']) == len(_nested_score['child']) == len(_nested_score['index'])
                best_score = min(_nested_score['score'])
                best_i = _nested_score['index'][_nested_score['score'].index(best_score)]
                return decode_score(_nested_score['child'][best_i], total_score + best_score)

        data_loader, partition, data_dk = self.batch_encode_plus_mask(
            texts,
            batch_token_to_mask=tokens_to_mask,
            batch_size=batch_size)
        score = self.__get_nll(data_loader, reduce=False)
        conditional_nll = list(map(
            lambda x: decode_score(x[0].restore_structure(score[x[1][0]:x[1][1]], insert_key='score')),
            zip(data_dk, partition)
        ))
        if tokens_to_condition:
            data_loader, partition, data_dk = self.batch_encode_plus_mask(
                texts,
                batch_token_to_mask=tokens_to_mask,
                batch_token_to_mask_no_label=tokens_to_condition,
                batch_size=batch_size)
            score = self.__get_nll(data_loader, reduce=False)
            marginal_nll = list(map(
                lambda x: decode_score(x[0].restore_structure(score[x[1][0]:x[1][1]], insert_key='score')),
                zip(data_dk, partition)
            ))
            assert len(conditional_nll) == len(marginal_nll)
            negative_pmi = list(map(lambda x: x[0] * weight - x[1], zip(conditional_nll, marginal_nll)))
            return negative_pmi
        else:
            return conditional_nll

    ######################################
    # Modules for perplexity computation #
    ######################################
    def encode_plus_perplexity(self, text: str):
        """ An output from `encode_plus` for perplexity computation
        * for pseudo perplexity, encode all text with mask on every token one by one
        :param str text: a text to encode
        :return a list of encode
        """
        assert type(text) is str
        assert len(text.replace(' ', '')) != 0, 'found an empty text'
        param = {'max_length': self.max_length, 'truncation': True, 'padding': 'max_length'}

        if self.is_causal:
            encode = self.tokenizer.encode_plus(text, **param)
            encode['labels'] = self.input_ids_to_labels(encode['input_ids'])
            return [encode]
        else:
            token_list = self.tokenizer.tokenize(text)

            def encode_with_single_mask_id(mask_position: int):
                _token_list = token_list.copy()  # can not be encode outputs because of prefix
                masked_token_id = self.tokenizer.convert_tokens_to_ids(_token_list[mask_position])
                _token_list[mask_position] = self.tokenizer.mask_token
                _encode = self.tokenizer.encode_plus(_token_list, **param)
                _encode['labels'] = self.input_ids_to_labels(
                    _encode['input_ids'],
                    label_position=[mask_position + len(self.sp_token_prefix)],
                    label_id=[masked_token_id])
                return _encode

            return [encode_with_single_mask_id(i) for i in range(len(token_list))]

    def batch_encode_plus_perplexity(self, batch_text: List, batch_size: int = None):
        """ Batch version of `self.encode_plus_perplexity`

        :param batch_text: a batch of `text`
        :param batch_size: batch size
        :return: (`torch.utils.data.DataLoader` class, partition)
        """
        batch_size = len(batch_text) if batch_size is None else batch_size
        # for x in tqdm(zip(batch_text, batch_token_to_mask, batch_token_to_mask_no_label)):
        #     data_dk.append(self.encode_plus_mask(text=x[0], token_to_mask=x[1], token_to_mask_no_label=x[2]))
        logging.info('creating data loader')
        # data = list(map(lambda x: self.encode_plus_perplexity(x), batch_text))
        data = []
        for x in tqdm(batch_text):
            data.append(self.encode_plus_perplexity(x))

        partition = get_partition(data)

        data_loader = torch.utils.data.DataLoader(
            Dataset(list(chain(*data))),
            num_workers=self.num_worker, batch_size=batch_size, shuffle=False, drop_last=False
        )
        return data_loader, partition

    def get_perplexity(self, texts: List, batch_size: int = None):
        """ (pseudo) Perplexity

        :param texts:
        :param batch_size:
        :return: a list of (pseudo) perplexity
        """
        assert type(texts) is list, 'type error'
        if not self.model:
            self.load_model()

        data_loader, partition = self.batch_encode_plus_perplexity(texts, batch_size=batch_size)
        nll = self.__get_nll(data_loader)
        # for pseudo likelihood aggregation
        return list(map(lambda x: math.exp(sum(nll[x[0]:x[1]]) / (x[1] - x[0])), partition))

    ###################################
    # Modules for embedding operation #
    ###################################
    def encode_plus_embedding(self, text: str, token_to_embed: List):
        """

        :param text:
        :param token_to_embed:
        :return:
        """

        assert type(text) is str
        assert len(text.replace(' ', '')) != 0, 'found an empty text'

        param = {'max_length': self.max_length, 'truncation': True, 'padding': 'max_length'}
        encode = self.tokenizer.encode_plus(text, **param)

        token_list = self.tokenizer.tokenize(text)
        positions = [self.find_position(s, text, token_list) for s in token_to_embed]
        pos = [[len(self.sp_token_prefix) + s, len(self.sp_token_prefix) + e] for s, e in positions]
        assert len(pos) == 4, 'token_to_embed is allowed upto 4'
        # encode['position_to_embed'] = pos + [[0, 0]] * (5 - len(pos))
        encode['position_to_embed'] = pos
        return encode

    def batch_encode_plus_embedding(self,
                                    batch_text: List,
                                    batch_token_to_embed: List,
                                    batch_size: int = None):
        batch_size = len(batch_text) if batch_size is None else batch_size
        data = list(map(lambda x: self.encode_plus_embedding(*x), zip(batch_text, batch_token_to_embed)))
        return torch.utils.data.DataLoader(
            Dataset(data),
            num_workers=self.num_worker, batch_size=batch_size, shuffle=False, drop_last=False
        )

    @staticmethod
    def relation_similarity(embedding_tensor, positions):
        """ Get relation similarity

        :param embedding_tensor: a tensor (length, dim)
        :param positions: a list of position (start, end), which should contain 4 different
        :return:
        """

        def cos_similarity(a: List, b: List):
            assert len(a) == len(b)
            norm_a = sum(map(lambda x: x * x, a)) ** 0.5
            norm_b = sum(map(lambda x: x * x, b)) ** 0.5
            inner_prod = sum(map(lambda x: x[0] * x[1], zip(a, b)))
            return inner_prod / (norm_a * norm_b)

        word_embedding = list(map(lambda x: torch.mean(embedding_tensor[x[0]:x[1]], 0).cpu().tolist(), positions))
        assert len(word_embedding) == 4
        diff_stem = list(map(lambda x: x[0] - x[1], word_embedding))
        diff_predict = list(map(lambda x: x[2] - x[3], word_embedding))
        return cos_similarity(diff_stem, diff_predict)

    def get_embedding_similarity(self,
                                 texts: List,
                                 tokens_to_embed: List, batch_size: int = None):
        """ Similarity of embedding differences over relations

        :param texts:
        :param tokens_to_embed:
        :param batch_size:
        :return: embeddings (len(texts), token num, dim)
        """
        assert type(texts) is list, 'type error'
        if not self.model:
            self.load_model(lm_head=False)

        data_loader = self.batch_encode_plus_embedding(
            texts, batch_size=batch_size, batch_token_to_embed=tokens_to_embed)

        embeddings = []

        with torch.no_grad():
            for encode in tqdm(data_loader):
                position_to_embed = encode.pop('position_to_embed').cpu().tolist()
                encode = {k: v.to(self.device) for k, v in encode.items()}
                output = self.model(**encode, return_dict=True)
                last_hidden_state = output['last_hidden_state']  # batch, length, dim
                embeddings += list(map(
                    lambda n: self.relation_similarity(last_hidden_state[n], position_to_embed[n]),
                    range(len(position_to_embed))
                ))

        return embeddings
