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
    """ Prompt generator based on pretrained language models """

    def __init__(self,
                 model: str,
                 max_length: int = 32,
                 cache_dir: str = None,
                 num_worker: int = 0):
        """ Prompt generator based on pretrained language models

        :param model: a model name corresponding to a model card in `transformers`
        :param max_length: a model max length if specified, else use model_max_length
        """
        logging.debug('*** setting up a language model ***')
        assert 'bert' in model, '{} is not BERT'.format(model)
        self.num_worker = num_worker
        self.model_name = model
        self.cache_dir = cache_dir
        self.device = None
        self.model = None
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)
        self.config = transformers.AutoConfig.from_pretrained(model, cache_dir=cache_dir)
        self.max_length = self.tokenizer.model_max_length
        if max_length:
            assert self.max_length >= max_length, '{} < {}'.format(self.max_length, max_length)
            self.max_length = max_length

        # sentence prefix tokens to fix offset when encoding sentence
        tokens = self.tokenizer.tokenize('get tokenizer specific prefix')
        tokens_encode = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode('get tokenizer specific prefix'))
        self.sp_token_prefix = tokens_encode[:tokens_encode.index(tokens[0])]

    def __load_model(self):
        """ Load pretrained language model """
        if self.model:
            return
        logging.info('loading language model')
        self.model = transformers.AutoModelForMaskedLM.from_pretrained(
            self.model_name, config=self.config, cache_dir=self.cache_dir)
        self.model.eval()
        # GPU setup
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        self.model.to(self.device)
        logging.info('running on {} GPU'.format(torch.cuda.device_count()))

    def input_ids_to_labels(self,
                            input_ids,
                            label_position: List = None,
                            label_id: List = None):
        """ Generate a label for language model loss. If `label_position` is None, label is the original input_ids for
        every token except padding token, or it keeps what specified by `label_position` with label defined by 
        `label_id`.

        :param input_ids: input_ids given by tokenizer.encode
        :param label_position: (optional) position in input_ids for prediction
        :param label_id: (optional) token id for each position in `label_position`
        :return: `label` that can be used for loss computation
        """
        if label_position is None and label_id is None:  # just to ignore padding token
            label = list(map(lambda x: PAD_TOKEN_LABEL_ID if x == self.tokenizer.pad_token_id else x, input_ids))
        else:
            assert len(label_position) == len(label_id), '{} != {}'.format(len(label_position), len(label_id))
            label = [PAD_TOKEN_LABEL_ID] * len(input_ids)
            for p, i in zip(label_position, label_id):
                label[p] = i
        return label

    def encode_plus(self,
                    sentence: str,
                    token_wise_mask: bool = False):
        """ Encoding sentence with label that is
        - masked token if `token_wise_mask` is False (mainly for token prediction)
        - otherwise every token that is not mask token (mainly for perplexity computation)

        :param sentence: a string sentence
        :param token_wise_mask: (optional) to encode the sentence with a mask on each position
        :return: a list of the output from tokenizer.encode_plus
        """
        param = {'max_length': self.max_length, 'truncation': True, 'padding': 'max_length'}
        if not token_wise_mask:
            assert self.tokenizer.mask_token in sentence, 'sentence has no masks: {}'.format(sentence)
            encode = self.tokenizer.encode_plus(sentence, **param)
            assert encode['input_ids'][-1] == self.tokenizer.pad_token_id, 'exceeded max_length'
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
                assert _encode['input_ids'][-1] == self.tokenizer.pad_token_id, 'exceeded max_length'
                _encode['labels'] = self.input_ids_to_labels(
                    _encode['input_ids'],
                    label_position=[mask_position + len(self.sp_token_prefix)],
                    label_id=[masked_token_id])
                return _encode

            length = min(self.max_length - len(self.sp_token_prefix), len(token_list))
            return list(filter(None, map(encode_with_single_mask_id, range(length))))

    def cleanup_decode(self, sentence):
        """ Clean up sentence with toknizers special tokens """ 
        mask = self.tokenizer.mask_token
        # give a space around mask token
        cleaned_sent = re.sub(r'({})'.format(mask).replace('[', '\[').replace(']', '\]'), r' \1 ', sentence)
        # reduce more than two spaces to one
        cleaned_sent = re.sub(r'\s+', ' ', cleaned_sent)
        # remove special tokens but keep mask
        to_remove = list(filter(lambda x: x != mask, self.tokenizer.all_special_tokens))
        to_remove = '|'.join(to_remove).replace('[', '\[').replace(']', '\]')
        cleaned_sent = re.sub(r'{}'.format(to_remove), '', cleaned_sent)
        # remove redundant spaces on the beginning of the sentence
        return re.sub(r'\A\s*', '', cleaned_sent)

    def pair_to_seed(self,
                     word_pair: List,
                     n_blank: int = 3,
                     n_blank_b: int = 2,
                     n_blank_e: int = 2,
                     seed_type: str = 'middle'):
        """ Convert word pair to a seed template with placeholders by masking token

        :param word_pair: a list of two words
        :param n_blank: the number of mask in between word_pair
        :param n_blank_b: the number of mask at the beginning of the template
        :param n_blank_e: the number of mask at the end of the template
        :param seed_type: template type where
            'middle': word_pair[0] + ['<mask'] * n_blank + word_pair[1]
            'whole': ['<mask'] * n_blank_b + word_pair[0] + ['<mask'] * n_blank + word_pair[1] + ['<mask'] * n_blank_e
        :return: the generated template
        """
        assert len(word_pair) == 2, 'word_pair contains wrong number of tokens: {}'.format(len(word_pair))
        mask = self.tokenizer.mask_token
        h, t = word_pair
        if seed_type == 'middle':
            return ' '.join([h] + [mask] * n_blank + [t])
        elif seed_type == 'whole':
            return ' '.join([mask] * n_blank_b + [h] + [mask] * n_blank + [t] + [mask] * n_blank_e)
        else:
            raise ValueError('unknown seed type: {}'.format(seed_type))

    def generate(self,
                 word_pairs: List = None,
                 seed_sentences: List = None,
                 n_revision: int = 10,
                 topk: int = 10,
                 batch_size: int = 4,
                 debug: bool = False,
                 seed_type: str = 'middle',
                 n_blank: int = 4,
                 n_blank_b: int = 1,
                 n_blank_e: int = 1):
        """ Generate/rewrite prompt based on perplexity

        :param word_pairs: a list of two words
        :param seed_sentences: a list of sentences
        :param n_revision: the number of revision after replacing all the mask
        :param batch_size: batch size
        :param topk: see Prompter.replace_single_token
        :param debug: see Prompter.replace_single_token
        :param seed_type: see Prompter.pair_to_seed
        :param n_blank: see Prompter.pair_to_seed
        :param n_blank_b: see Prompter.pair_to_seed
        :param n_blank_e: see Prompter.pair_to_seed
        :return: 
        """
        if seed_sentences:
            assert not word_pairs, 'both of `seed_sentences` and `word_pairs` are given'
            if type(seed_sentences) is str:
                seed_sentences = [seed_sentences]
            edit = [[s] for s in seed_sentences]
            data_key = {k: v for k, v in enumerate(seed_sentences)}
        else:
            assert word_pairs, 'either of `seed_sentences` or `word_pairs` is required'
            if type(word_pairs[0]) is not list:
                word_pairs = [word_pairs]
            seed_sentences = list(map(lambda x: self.pair_to_seed(
                x, seed_type=seed_type, n_blank=n_blank, n_blank_b=n_blank_b, n_blank_e=n_blank_e), word_pairs))
            data_key = {k: '||'.join(v) for k, v in enumerate(word_pairs)}
        logging.info('\n################\n# REPLACE MASK #\n################')
        edit = [seed_sentences]
        edit_ppl = []
        while True:
            if any(map(lambda x: self.tokenizer.mask_token not in x, seed_sentences)):
                # mask should be removed one by one, but some has skipped if this raises error
                assert all(self.tokenizer.mask_token not in i for i in seed_sentences), 'some masks got lost'
                break
            logging.info('REPLACE MASK: step {}'.format(len(edit_ppl)))
            seed_sentences, ppl = self.replace_single_token(
                seed_sentences, vocab_to_keep=word_pairs, topk=topk, debug=debug, batch_size=batch_size)
            edit.append(seed_sentences)
            edit_ppl.append(ppl)
        edit = list(zip(*edit))
        edit_ppl = [[]] * len(seed_sentences) if len(edit_ppl) == 0 else list(zip(*edit_ppl))

        logging.info('\n######################\n# ITERATIVE REVISION #\n######################')
        output_dict = {}
        data_index = list(data_key.keys())
        for i in range(n_revision):
            logging.info('ITERATIVE REVISION: step {}/{}'.format(i, n_revision))
            seed_sentences, ppl = self.replace_single_token(
                seed_sentences, vocab_to_keep=word_pairs, topk=topk, debug=debug, batch_size=batch_size)

            # extract stable sentence
            index_fixed = list(filter(lambda x: seed_sentences[x] == edit[x][-1], range(len(seed_sentences))))
            output_dict.update({data_key[data_index[n]]: [edit[n], edit_ppl[n]] for n in index_fixed})

            # sentence keep improving
            index_unfixed = list(filter(lambda x: seed_sentences[x] != edit[x][-1], range(len(seed_sentences))))
            edit = list(map(lambda x: tuple(list(edit[x]) + [seed_sentences[x]]), index_unfixed))
            edit_ppl = list(map(lambda x: tuple(list(edit_ppl[x]) + [ppl[x]]), index_unfixed))

            seed_sentences = list(map(lambda x: seed_sentences[x], index_unfixed))
            ppl = list(map(lambda x: ppl[x], index_unfixed))
            data_index = list(map(lambda x: data_index[x], index_unfixed))
            if word_pairs:
                word_pairs = list(map(lambda x: word_pairs[x], index_unfixed))

            if len(seed_sentences) == 0:
                logging.info('ITERATIVE REVISION: all sentences reached the best perplexity')
                break

        output_dict.update({data_key[i]: [edit[i], edit_ppl[i]] for n, i in enumerate(data_index)})
        return output_dict

    def replace_single_token(self,
                             seed_sentences: List,
                             vocab_to_keep: List = None,
                             batch_size: int = 4,
                             topk: int = 5,
                             debug: bool = False):
        """ Replace single token (run parallel over given lists)
        - (i) Greedy token prediction: predict token by masking each token or masked token if sentence consists of mask
        - (ii) Perplexity re-ranking: choose the best replaced sentence that achieves the best perplexity

        :param seed_sentences: a list of sentence
        :param vocab_to_keep: (optional) a list of token to keep while replacing
        # :param vocab_to_keep_unique: (optional) limit the tokens from vocab_to_keep be unique while replacing
        :param batch_size: batch size
        :param topk: keep topk prediction on masked token for perplexity filtering
        :param debug: to show a few replaced sentence
        :return:
        """
        if vocab_to_keep:
            assert len(seed_sentences) == len(vocab_to_keep), '{} != {}'.format(len(seed_sentences), len(vocab_to_keep))
        topk_per_position = 100
        self.__load_model()
        if type(seed_sentences) is str:
            seed_sentences = [seed_sentences]

        # if mask is in sentence, it is first replaced, otherwise all tokens are considered
        data = map(lambda x: self.encode_plus(x, token_wise_mask=self.tokenizer.mask_token not in x), seed_sentences)
        data = list(data)
        partition = get_partition(data)
        data_loader = torch.utils.data.DataLoader(
            Dataset(list(chain(*data))),
            num_workers=self.num_worker, batch_size=batch_size, shuffle=False, drop_last=False)

        logging.info('\t* prediction on masked tokens')
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
        logging.info('\t* filter to top {} prediction'.format(topk))
        for partition_n, (s, e) in enumerate(tqdm(partition)):

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
                        decoded_no_mask = decoded.replace(self.tokenizer.mask_token, '')
                        # check if all tokens from keep_vocab in the decoded sentence
                        if vocab_to_keep is None:
                            count = [1]
                        elif allow_subword:
                            count = list(map(lambda x: len(re.findall(x, decoded_no_mask)), vocab_to_keep[partition_n]))
                        else:
                            count = list(map(lambda x: len(re.findall(r'\b{}\b'.format(x), decoded_no_mask)),
                                             vocab_to_keep[partition_n]))
                        if not all(count):
                            return None

                        # check if all tokens from keep_vocab just appeared once
                        # if vocab_to_keep_unique:
                        if not all(map(lambda x: x == 1, count)):
                            return None
                        return decoded, token_likelihood[k]

                    for _replace_pos, (_val, _ind) in filtered:
                        topk_decoded += list(filter(
                            None, map(lambda x: decode_topk(x, _replace_pos, _ind, _val), range(_topk))
                        ))
                return topk_decoded

            topk_edit = process_single_pair(topk)
            if len(topk_edit) == 0 and topk_per_position > topk:
                topk_edit = process_single_pair(topk_per_position)
            if len(topk_edit) == 0:
                topk_edit = process_single_pair(topk_per_position, True)
                if len(topk_edit) != 0 and vocab_to_keep is not None:
                    logging.warning('prompt may include subword: `{}` ({})'.format(
                        topk_edit[0], vocab_to_keep[partition_n]))

            if len(topk_edit) == 0:
                raise ValueError('no valid sentence found: ({})\n- current prompt: {}'.format(
                    vocab_to_keep[partition_n], seed_sentences[partition_n]))
            # drop duplicated decode and keep the one with tje highest likelihood
            topk_edit = list(map(
                lambda d: max(filter(lambda x: x[0] == d, topk_edit), key=lambda x: x[1]),
                set(list(zip(*topk_edit))[0])
            ))
            topk_edit = sorted(topk_edit, key=lambda x: x[1], reverse=True)
            greedy_filling.append(list(zip(*topk_edit))[0][:min(topk, len(topk_edit))])

        logging.info('\t* ppl re-ranking')
        partition = get_partition(greedy_filling)
        list_ppl = self.get_perplexity(list(chain(*greedy_filling)), batch_size=batch_size)
        list_ppl = [list_ppl[s:e] for s, e in partition]
        best_edit = []
        best_ppl = []
        for sent, ppl in zip(greedy_filling, list_ppl):
            best_edit.append(sent[ppl.index(min(ppl))])
            best_ppl.append(min(ppl))

        if debug:
            logging.info('\t* edit sample')
            for n, (o, ed, bp) in enumerate(zip(seed_sentences, best_edit, best_ppl)):
                logging.info('\t\t- original: {}'.format(o))
                logging.info('\t\t- edit    : {} (ppl: {})'.format(ed, bp))
                if n > 5:
                    break
        return best_edit, best_ppl

    def get_perplexity(self, sentences, batch_size: int = 4):
        """ Compute perplexity on sentences

        :param batch_size:
        :param sentences: a list of strings
        :return: a list of perplexity
        """
        self.__load_model()
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
        return list(map(lambda x: math.exp(sum(nll[x[0]:x[1]]) / (x[1] - x[0])), partition))


if __name__ == '__main__':
    from pprint import pprint
    # lm = Prompter('albert-base-v1', max_length=16)
    lm = Prompter('roberta-base', max_length=16)

    # test_candidates = [["pleasure", "hedonist"], ["emotion", "demagogue"], ["opinion", "sympathizer"]]
    # pprint(lm.generate(word_pairs=test_candidates, debug=True))

    test_sentences = ['emotion and violence: demagogue', 'opinion of a communist sympathizer']
    pprint(lm.generate(seed_sentences=test_sentences, debug=True))
