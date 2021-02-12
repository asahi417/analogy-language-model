import re
import os
import logging
import math
from itertools import chain
from typing import List
from tqdm import tqdm
from copy import deepcopy
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

import transformers
import torch
from torch import nn

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning message
PAD_TOKEN_LABEL_ID = nn.CrossEntropyLoss().ignore_index


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


class Prompter:
    """ transformers language model based sentence-mining """

    def __init__(self,
                 model: str,
                 max_length: int = None,
                 cache_dir: str = './cache',
                 num_worker: int = 0):
        """ transformers language model based sentence-mining

        :param model: a model name corresponding to a model card in `transformers`
        :param max_length: a model max length if specified, else use model_max_length
        """
        logging.debug('*** setting up a language model ***')
        self.num_worker = num_worker
        if self.num_worker == 1:
            os.environ["OMP_NUM_THREADS"] = "1"  # to turn off warning message

        self.model_type = None
        self.model_name = model
        self.cache_dir = cache_dir
        self.device = 'cpu'
        self.model = None
        self.is_causal = 'gpt' in self.model_name  # TODO: fix to be more comprehensive method
        assert not self.is_causal
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)
        if self.is_causal:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.config = transformers.AutoConfig.from_pretrained(model, cache_dir=cache_dir)
        if max_length:
            assert self.tokenizer.model_max_length >= max_length, '{} < {}'.format(self.tokenizer.model_max_length,
                                                                                   max_length)
            self.max_length = max_length
        else:
            self.max_length = self.tokenizer.model_max_length

        # sentence prefix tokens
        tokens = self.tokenizer.tokenize('get tokenizer specific prefix')
        tokens_encode = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode('get tokenizer specific prefix'))
        self.sp_token_prefix = tokens_encode[:tokens_encode.index(tokens[0])]
        self.sp_token_suffix = tokens_encode[tokens_encode.index(tokens[-1]) + 1:]

    def input_ids_to_labels(self, input_ids, label_position: List = None, label_id: List = None):
        """ Generate label for likelihood computation

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

    def cleanup_decode(self, sentence):
        # give a space around <mask>
        cleaned_sent = re.sub(r'({})'.format(self.tokenizer.mask_token), r' \1 ', sentence)
        cleaned_sent = re.sub(r'\s+', ' ', cleaned_sent)  # reduce more than two space to one

        # remove special tokens but mask
        to_remove = list(filter(lambda x: x != self.tokenizer.mask_token, self.tokenizer.all_special_tokens))
        to_remove = '|'.join(to_remove).replace('[', '\[').replace(']', '\]')
        cleaned_sent = re.sub(r'{}'.format(to_remove), '', cleaned_sent)

        # remove redundant spaces at the prefix
        return re.sub(r'\A\s*', '', cleaned_sent)

    def load_model(self):
        """ Model setup """
        logging.info('load language model')
        params = dict(config=self.config, cache_dir=self.cache_dir)
        if self.is_causal:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_name, **params)
            self.model_type = 'causal_lm'
        else:
            self.model = transformers.AutoModelForMaskedLM.from_pretrained(self.model_name, **params)
            self.model_type = 'masked_lm'
        self.model.eval()
        # gpu
        n_gpu = torch.cuda.device_count()
        assert n_gpu <= 1
        self.device = 'cuda' if n_gpu > 0 else 'cpu'
        self.model.to(self.device)
        logging.info('running on {} GPU'.format(n_gpu))

    def pair_to_seed(self,
                     word_pair: List,
                     n_blank: int = 3,
                     n_blank_prefix: int = 2,
                     n_blank_suffix: int = 2,
                     batch_size: int = 4,
                     seed_type: str = 'middle'):
        assert len(word_pair) == 2, '{}'.format(len(word_pair))
        h, t = word_pair
        if seed_type == 'middle':
            return ' '.join([h] + [self.tokenizer.mask_token] * n_blank + [t])
        elif seed_type == 'whole':
            return ' '.join([self.tokenizer.mask_token] * n_blank_prefix + [h] + [self.tokenizer.mask_token] * n_blank
                            + [t] + [self.tokenizer.mask_token] * n_blank_suffix)
        elif seed_type == 'best':
            # build candidates
            candidates = []
            for pre_n in range(self.max_length - 2):
                prefix = [self.tokenizer.mask_token] * pre_n + [h]
                for mid_n in range(1, self.max_length - 1 - pre_n):
                    middle = [self.tokenizer.mask_token] * mid_n + [t]
                    candidates.append(' '.join(prefix + middle))
            # compute perplexity
            logging.info('find best seed position for head and tail by perplexity: {} in total'.format(len(candidates)))
            ppl = self.get_perplexity(candidates, batch_size=batch_size)
            best_seed = candidates[ppl.index(min(ppl))]
            print(candidates)
            print(ppl)
            print(best_seed)
            return best_seed
        else:
            raise ValueError('unknown seed type: {}'.format(seed_type))

    def encode_plus(self, sentence):
        param = {'max_length': self.max_length, 'truncation': True, 'padding': 'max_length'}
        encode = self.tokenizer.encode_plus(sentence, **param)
        assert self.tokenizer.mask_token_id in encode['input_ids']
        encode['mask_flag'] = list(map(lambda x: int(x == self.tokenizer.mask_token_id), encode['input_ids']))
        return encode

    def replace_mask(self,
                     word_pairs: List,
                     n_blank: int = 3,
                     topk: int = 5,
                     seed_type: str = 'middle',
                     batch_size: int = 4,
                     perplexity_filter: bool = True,
                     debug: bool = False,
                     n_blank_prefix: int = 2,
                     n_blank_suffix: int = 2):
        if type(word_pairs[0]) is not list:
            word_pairs = [word_pairs]
        shared = {'n_blank': n_blank, 'seed_type': seed_type, 'n_blank_prefix': n_blank_prefix,
                  'n_blank_suffix': n_blank_suffix, 'batch_size': batch_size}
        seed_sentences = list(map(lambda x: self.pair_to_seed(x, **shared), word_pairs))
        shared = {'topk': topk, 'debug': debug, 'batch_size': batch_size, 'perplexity_filter': perplexity_filter}
        for i in range(n_blank):
            seed_sentences = self.replace_single_mask(seed_sentences, **shared)
        return seed_sentences

    def replace_single_mask(self,
                            seed_sentences,
                            batch_size: int = 4,
                            topk: int = 5,
                            perplexity_filter: bool = True,
                            debug: bool = False):
        if self.model is None:
            self.load_model()
        if type(seed_sentences) is str:
            seed_sentences = [seed_sentences]

        data = list(map(self.encode_plus, seed_sentences))
        data_loader = torch.utils.data.DataLoader(
            Dataset(data),
            num_workers=self.num_worker,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False)

        logging.info('Inference on masked token')
        total_input = []
        total_mask = []
        total_val = []  # batch, mask_size, topk
        total_ind = []
        with torch.no_grad():
            for encode in tqdm(data_loader):
                encode = {k: v.to(self.device) for k, v in encode.items()}
                mask_flag = encode.pop('mask_flag')
                output = self.model(**encode, return_dict=True)
                prediction_scores = output['logits']
                values, indices = prediction_scores.topk(topk, dim=-1)
                total_input += encode.pop('input_ids').tolist()
                total_mask += mask_flag.tolist()
                total_val += values.tolist()
                total_ind += indices.tolist()

        def edit_input(batch_i):
            inp, mas, val, ind = total_input[batch_i], total_mask[batch_i], total_val[batch_i], total_ind[batch_i]
            filtered = list(filter(lambda x: mas[x[0]] == 1, enumerate(zip(val, ind))))
            # to replace the position with the highest likelihood among possible masked positions
            replace_pos, (_, ind) = sorted(filtered, key=lambda x: x[1][0][0], reverse=True)[0]

            def decode_topk(k):
                inp_ = deepcopy(inp)
                inp_[replace_pos] = ind[k]
                decoded = self.tokenizer.decode(inp_, skip_special_tokens=False)
                return self.cleanup_decode(decoded)

            topk_decoded = list(map(decode_topk, range(topk)))
            return topk_decoded

        greedy_filling = list(map(edit_input, range(len(total_input))))
        if perplexity_filter:
            logging.info('ppl filtering')
            best_edit = []
            for s in greedy_filling:
                ppl = self.get_perplexity(s)
                best_edit.append(s[ppl.index(min(ppl))])
                # best_edit.append(s[ppl.index(max(ppl))])
        else:
            best_edit = list(map(lambda x: x[0], greedy_filling))

        if debug:
            for o, e in zip(seed_sentences, best_edit):
                logging.info('\n- original: {}\n- edit : {}\n'.format(o, e))
        return best_edit

    def encode_plus_perplexity(self, sentence):
        """ An output from `encode_plus` for perplexity computation
        * for pseudo perplexity, encode all text with mask on every token one by one
        """
        param = {'max_length': self.max_length, 'truncation': True, 'padding': 'max_length'}
        if self.is_causal:
            raise NotImplementedError('TODO')
        else:
            token_list = self.tokenizer.tokenize(sentence)

            def encode_with_single_mask_id(mask_position: int):
                _token_list = token_list.copy()  # can not be encode outputs because of prefix
                masked_token_id = self.tokenizer.convert_tokens_to_ids(_token_list[mask_position])
                if masked_token_id == self.tokenizer.mask_token_id:
                    return None
                _token_list[mask_position] = self.tokenizer.mask_token
                tmp_string = self.tokenizer.convert_tokens_to_string(_token_list)
                _encode = self.tokenizer.encode_plus(tmp_string, **param)
                _encode['labels'] = self.input_ids_to_labels(
                    _encode['input_ids'],
                    label_position=[mask_position + len(self.sp_token_prefix)],
                    label_id=[masked_token_id])
                return _encode

            length = min(self.max_length - len(self.sp_token_prefix), len(token_list))
            return list(filter(None, map(encode_with_single_mask_id, range(length))))

    def get_perplexity(self, sentences, batch_size: int = 4):
        """ compute perplexity on each sentence

        :param batch_size:
        :param sentences:
        :return: a list of perplexity
        """
        if self.model is None:
            self.load_model()
        if type(sentences) is str:
            sentences = [sentences]

        data = list(map(self.encode_plus_perplexity, sentences))
        partition = get_partition(data)

        data_loader = torch.utils.data.DataLoader(
            Dataset(list(chain(*data))),
            num_workers=self.num_worker, batch_size=batch_size, shuffle=False, drop_last=False
        )
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
                loss = torch.sum(loss, -1)
                nll += list(map(
                    lambda x: x[0] / sum(map(lambda y: y != PAD_TOKEN_LABEL_ID, x[1])),
                    zip(loss.cpu().tolist(), labels.cpu().tolist())
                ))
        perplexity = list(map(lambda x: math.exp(sum(nll[x[0]:x[1]]) / (x[1] - x[0])), partition))
        return perplexity


if __name__ == '__main__':
    lm = Prompter('roberta-large', max_length=12)
    # stem = ["beauty", "aesthete"]
    candidates_ = [["pleasure", "hedonist"], ["emotion", "demagogue"], ["opinion", "sympathizer"],
                   ["seance", "medium"], ["luxury", "ascetic"]]
    o_ = lm.replace_mask(candidates_,
                         seed_type='best',
                         debug=True,
                         perplexity_filter=True,
                         topk=5)
    print(o_)
