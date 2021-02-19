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
                 max_length: int = 32,
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
        cleaned_sent = re.sub(r'({})'.format(self.tokenizer.mask_token).replace('[', '\[').replace(']', '\]'),
                              r' \1 ', sentence)
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
        # elif seed_type == 'best':
        #     # build candidates
        #     candidates = []
        #     for pre_n in range(self.max_length - 2):
        #         prefix = [self.tokenizer.mask_token] * pre_n + [h]
        #         for mid_n in range(1, self.max_length - 1 - pre_n):
        #             middle = [self.tokenizer.mask_token] * mid_n + [t]
        #             candidates.append(' '.join(prefix + middle))
        #     # compute perplexity
        #     logging.info('find best seed position for head and tail by perplexity: {} in total'.format(len(candidates)))
        #     ppl = self.get_perplexity(candidates, batch_size=batch_size)
        #     best_seed = candidates[ppl.index(min(ppl))]
        #     return best_seed
        else:
            raise ValueError('unknown seed type: {}'.format(seed_type))

    def encode_plus(self,
                    sentence,
                    token_wise_mask: bool = False):
        """ Encode with mask flag, that is masked position if sentence has masked token, otherwise is the entire
        sequence except for special tokens.

        :param sentence:
        :param token_wise_mask:
        :return:
        """
        param = {'max_length': self.max_length, 'truncation': True, 'padding': 'max_length'}
        if self.is_causal:
            raise NotImplementedError('only available with masked LM')
        if not token_wise_mask:
            assert self.tokenizer.mask_token in sentence, sentence
            encode = self.tokenizer.encode_plus(sentence, **param)
            assert encode['input_ids'][-1] == self.tokenizer.pad_token_id, 'exceed max_length'
            return [encode]
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
                assert _encode['input_ids'][-1] == self.tokenizer.pad_token_id, 'exceed max_length'
                _encode['labels'] = self.input_ids_to_labels(
                    _encode['input_ids'],
                    label_position=[mask_position + len(self.sp_token_prefix)],
                    label_id=[masked_token_id])
                return _encode

            length = min(self.max_length - len(self.sp_token_prefix), len(token_list))
            return list(filter(None, map(encode_with_single_mask_id, range(length))))

    def replace_mask(self,
                     word_pairs: List,
                     n_blank: int = 4,
                     n_revision: int = 10,
                     topk: int = 10,
                     topk_per_position: int = 1000,
                     seed_type: str = 'middle',
                     batch_size: int = 4,
                     debug: bool = False,
                     no_repetition: bool = False,
                     n_blank_prefix: int = 1,
                     n_blank_suffix: int = 1):
        if type(word_pairs[0]) is not list:
            word_pairs = [word_pairs]
        shared = {'n_blank': n_blank, 'seed_type': seed_type, 'n_blank_prefix': n_blank_prefix,
                  'n_blank_suffix': n_blank_suffix, 'batch_size': batch_size}
        seed_sentences = list(map(lambda x: self.pair_to_seed(x, **shared), word_pairs))
        shared = {'word_pairs': word_pairs, 'topk': topk, 'topk_per_position': topk_per_position, 'debug': debug,
                  'batch_size': batch_size, 'no_repetition': no_repetition}
        logging.info('\n################\n# REPLACE MASK #\n################')
        edit = [seed_sentences]
        edit_ppl = []
        while True:
            logging.info('REPLACE MASK: step {}'.format(len(edit_ppl)))
            seed_sentences, ppl = self.replace_single_mask(seed_sentences, **shared)
            edit.append(seed_sentences)
            edit_ppl.append(ppl)
            if any(self.tokenizer.mask_token not in i for i in seed_sentences):
                # mask should be removed one by one, but some has skipped if this raises error
                assert all(self.tokenizer.mask_token not in i for i in seed_sentences), 'some masks got lost'
                break
        edit = list(zip(*edit))
        edit_ppl = list(zip(*edit_ppl))
        output_dict = {}
        if n_revision != 0:
            logging.info('\n#####################\n# PERPLEXITY FILTER #\n#####################')
            logging.info('PERPLEXITY FILTER: max {} steps'.format(n_revision))
            shared = {'topk': topk, 'topk_per_position': topk_per_position, 'debug': debug, 'batch_size': batch_size,
                      'no_repetition': no_repetition}
            for i in range(n_revision):
                logging.info('PERPLEXITY FILTER: step {}/{}'.format(i, n_revision))
                if len(seed_sentences) == 0:
                    logging.info('PERPLEXITY FILTER: all sentences reached the best perplexity')
                    break
                seed_sentences, ppl = self.replace_single_mask(seed_sentences, word_pairs=word_pairs, **shared)

                index_fixed = list(filter(lambda x: seed_sentences[x] == edit[x][-1], range(len(seed_sentences))))
                # extract stable sentence
                for n in index_fixed:
                    output_dict['||'.join(word_pairs[n])] = [edit[n], edit_ppl[n]]

                # sentence keep improving
                index_unfixed = list(filter(lambda x: seed_sentences[x] != edit[x][-1], range(len(seed_sentences))))
                seed_sentences = list(map(lambda x: seed_sentences[x], index_unfixed))
                ppl = list(map(lambda x: ppl[x], index_unfixed))
                word_pairs = list(map(lambda x: word_pairs[x], index_unfixed))
                edit = list(map(lambda x: edit[x], index_unfixed))
                edit_ppl = list(map(lambda x: edit_ppl[x], index_unfixed))

                for n in range(len(index_unfixed)):
                    edit[n] = tuple(list(edit[n]) + [seed_sentences[n]])
                    edit_ppl[n] = tuple(list(edit_ppl[n]) + [ppl[n]])

        output_dict_remains = {
            '||'.join(pair): [edit[n], edit_ppl[n]] for n, pair in enumerate(word_pairs)
        }
        output_dict.update(output_dict_remains)
        return output_dict

    def replace_single_mask(self, seed_sentences, word_pairs, batch_size: int = 4, topk: int = 5,
                            topk_per_position: int = 5, debug: bool = False,
                            no_repetition: bool = False):
        assert len(seed_sentences) == len(word_pairs), '{} != {}'.format(len(seed_sentences), len(word_pairs))
        if self.model is None:
            self.load_model()
        if type(seed_sentences) is str:
            seed_sentences = [seed_sentences]

        # sentence without masked token will perform token wise mask
        data = list(map(
            lambda x: self.encode_plus(x, token_wise_mask=self.tokenizer.mask_token not in x), seed_sentences))
        partition = get_partition(data)
        data_loader = torch.utils.data.DataLoader(
            Dataset(list(chain(*data))),
            num_workers=self.num_worker, batch_size=batch_size, shuffle=False, drop_last=False)
        assert len(word_pairs) == len(partition), '{} != {}'.format(len(word_pairs), len(partition))

        logging.info(' * prediction on masked tokens')
        total_input, total_val, total_ind = [], [], []
        with torch.no_grad():
            for encode in tqdm(data_loader):
                encode = {k: v.to(self.device) for k, v in encode.items()}
                output = self.model(**encode, return_dict=True)
                prediction_scores = output['logits']
                values, indices = prediction_scores.topk(topk_per_position, dim=-1)
                total_input += encode.pop('input_ids').tolist()
                total_val += values.tolist()
                total_ind += indices.tolist()

        greedy_filling = []
        logging.info(' * filter to top {} prediction'.format(topk))
        for partition_n, (s, e) in enumerate(tqdm(partition)):
            head, tail = word_pairs[partition_n]

            def process_single_pair(_topk, allow_subword=False):
                topk_decoded = []
                for i in range(s, e):
                    inp, val, ind = total_input[i], total_val[i], total_ind[i]
                    filtered = list(filter(
                        lambda x: inp[x[0]] == self.tokenizer.mask_token_id, enumerate(zip(val, ind))))

                    def decode_topk(k, replace_pos, token_index, token_likelihood):
                        tokens = deepcopy(inp)
                        tokens[replace_pos] = token_index[k]
                        decoded = self.tokenizer.decode(tokens, skip_special_tokens=False)
                        decoded = self.cleanup_decode(decoded)
                        print(decoded)
                        # skip if target word is not in the decoded (allow to be a subwword)
                        if allow_subword and head in decoded and tail in decoded:
                            if not no_repetition or (len(re.findall(r'\b{}\b'.format(head), decoded)) == 1
                                                     and len(re.findall(r'\b{}\b'.format(tail), decoded)) == 1):
                                return decoded, token_likelihood[k]
                        # skip if target word is replaced or merged into other words
                        if re.findall(r'\b{}\b'.format(head), decoded) and re.findall(r'\b{}\b'.format(tail), decoded):
                            if not no_repetition or (len(re.findall(r'\b{}\b'.format(head), decoded)) == 1
                                                     and len(re.findall(r'\b{}\b'.format(tail), decoded)) == 1):
                                return decoded, token_likelihood[k]
                        return None

                    for _replace_pos, (_val, _ind) in filtered:
                        topk_decoded += list(filter(
                            None, map(lambda x: decode_topk(x, _replace_pos, _ind, _val), range(_topk))
                        ))
                return topk_decoded

            topk_edit = process_single_pair(topk)
            if len(topk_edit) == 0:
                topk_edit = process_single_pair(topk_per_position)
            if len(topk_edit) == 0:
                topk_edit = process_single_pair(topk_per_position, True)
                if len(topk_edit) != 0:
                    logging.warning('prompt may include subword: `{}` ({}, {})'.format(topk_edit[0], head, tail))

            if len(topk_edit) == 0:
                raise ValueError('no valid sentence found: ({}, {})\n- current prompt: {}'.format(
                    head, tail, seed_sentences[partition_n]))
            # drop duplicated decode and keep the one with tje highest likelihood
            topk_edit = list(map(
                lambda d: max(filter(lambda x: x[0] == d, topk_edit), key=lambda x: x[1]),
                set(list(zip(*topk_edit))[0])
            ))
            topk_edit = sorted(topk_edit, key=lambda x: x[1], reverse=True)
            greedy_filling.append(list(zip(*topk_edit))[0][:min(topk, len(topk_edit))])

        # greedy_filling = list(map(process_single_sentence, tqdm(range(len(partition)))))
        logging.info(' * ppl filtering')
        partition = get_partition(greedy_filling)
        list_ppl = self.get_perplexity(list(chain(*greedy_filling)), batch_size=batch_size)
        list_ppl = [list_ppl[s:e] for s, e in partition]
        best_edit = []
        best_ppl = []
        for sent, ppl in zip(greedy_filling, list_ppl):
            best_edit.append(sent[ppl.index(min(ppl))])
            best_ppl.append(min(ppl))

        if debug:
            logging.info(' * edit sample')
            for n, (o, ed, bp) in enumerate(zip(seed_sentences, best_edit, best_ppl)):
                logging.info('  - original: {}'.format(o))
                logging.info('  - edit    : {} (ppl: {})'.format(ed, bp))
                if n > 5:
                    break
        return best_edit, best_ppl

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

        data = list(map(lambda x: self.encode_plus(x, token_wise_mask=True), sentences))
        partition = get_partition(data)

        data_loader = torch.utils.data.DataLoader(
            Dataset(list(chain(*data))),
            num_workers=self.num_worker, batch_size=batch_size, shuffle=False, drop_last=False)
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
    from pprint import pprint
    # lm = Prompter('albert-base-v1', max_length=12)
    lm = Prompter('roberta-base', max_length=16)
    # stem = ["beauty", "aesthete"]
    candidates_ = [["pleasure", "hedonist"],
                   ["emotion", "demagogue"],
                   ["opinion", "sympathizer"]]
    # candidates_ = ["emotion", "demagogue"]
    out = lm.replace_mask(
        candidates_,
        no_repetition=True,
        batch_size=1,
        seed_type='middle',
        topk=5,
        n_blank=2,
        n_revision=2,
        debug=True)
    pprint(out)
