""" pre-trained LM-based token revision """
import os
import logging
import math

from typing import List, Dict
from logging.config import dictConfig
from itertools import groupby

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
    """ simple torch.utils.data.Dataset instance to convert into tensors """
    float_tensors = ['attention_mask']

    def __init__(self, data: List):
        self.data = data

    def __len__(self):
        return len(self.data)

    def to_tensor(self, name, data):
        if name in self.float_tensors:
            return torch.tensor(data, dtype=torch.float32)
        return torch.tensor(data, dtype=torch.long)

    def __getitem__(self, idx):
        return {k: self.to_tensor(k, v) for k, v in self.data[idx].items()}


class TransformersLM:
    """ pre-trained Masked LM from huggingface transformers """

    def __init__(self, model: str):
        LOGGER.info('*** setting up language model ***')

        # model setup
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model, cache_dir=CACHE_DIR)
        self.sp_token_start, self.sp_token_sep, self.sp_token_end = get_special_tokens(self.tokenizer)
        self.config = transformers.AutoConfig.from_pretrained(model, cache_dir=CACHE_DIR)
        self.model = transformers.AutoModelForMaskedLM.from_pretrained(model, config=self.config, cache_dir=CACHE_DIR)
        self.model.eval()

        # gpu
        self.n_gpu = torch.cuda.device_count()
        assert self.n_gpu <= 1
        self.device = 'cuda' if self.n_gpu > 0 else 'cpu'
        self.model.to(self.device)
        LOGGER.info('running on %i GPUs' % self.n_gpu)

    def encode_plus_with_mask(self, text: str, token_to_mask: str):
        """ `encode_plus` with a token mask specified by string
        Note: it can only take single token, and a phrase over multiple tokens will raise error

        :param str text: a text to encode
        :param str token_to_mask: a target token to be masked
        :return encode, mask_position, (masked_token, masked_token_id)
        """

        # tokenize with special symbols
        token = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))

        # get first token matched `mask_string`
        mask_positions = [n for n, t in enumerate(token) if token_to_mask in t]
        assert len(mask_positions) > 0, '{} not in tokens {}'.format(token_to_mask, token)
        mask_position = mask_positions[0]

        # mask token and revert to text, `the<mask> is ~` == `the <mask> is ~`
        masked_token = token[mask_position]
        token[mask_position] = self.tokenizer.mask_token
        masked_token_id = self.tokenizer.convert_tokens_to_ids(masked_token)

        # encode sentence
        encode = self.tokenizer.encode_plus(token)

        return encode, mask_position, (masked_token, masked_token_id)

    def get_log_likelihood(self, text: str, token_to_mask: str):
        encode, mask_position, (masked_token, masked_token_id) = self.encode_plus_with_mask(text, token_to_mask)
        prob_dist = self.predict_probability(encode)
        prob_of_mask = prob_dist[mask_position]

        # log-likelihood
        likeli = math.log(prob_of_mask[masked_token_id])
        return likeli

    def predict_probability(self, encode: Dict):
        """ get probability distribution for given encoded input features """
        encode = {k: torch.tensor(v).to(self.device) if type(v) != torch.Tensor else v.to(self.device)
                  for k, v in encode.items()}
        logit = self.model(**encode)[0][0]
        prob_dist = torch.softmax(logit, dim=-1)
        return prob_dist

# def token_probability_batch(self, tokens: list, batch_size: int = 16):
    #     features = [self.tokenizer.encode_plus(i) for i in tokens]
    #     data_loader = torch.utils.data.DataLoader(
    #         Dataset(features), num_workers=NUM_WORKER, batch_size=batch_size, shuffle=False, drop_last=False)




    # def predict(self, tokens: list, id_to_mask: int, top_k_predict: int = 10):
    #     """ token prediction for masked token
    #
    #     :param tokens: list of token
    #     :param id_to_mask: token id to be masked
    #     :param top_k_predict: top k from prob distribution
    #     :return:
    #         predicted_token: list of top k prediction for masked token by the LM
    #         likelihood: log likelihood for masked token
    #     """
    #
    #     tokens = tokens.copy()
    #
    #     # for log-likelihood
    #     target_token = tokens[id_to_mask]
    #     target_token_id = self.tokenizer.convert_tokens_to_ids([target_token])[0]
    #     tokens[id_to_mask] = self.tokenizer.mask_token
    #
    #     # add bos symbol
    #     bos_length = 1
    #     if self.tokenizer.bos_token is not None:
    #         tokens = [self.tokenizer.bos_token] + tokens
    #     elif self.tokenizer.cls_token is not None:
    #         tokens = [self.tokenizer.cls_token] + tokens
    #     else:
    #         bos_length = 0
    #
    #     # add eos symbol
    #     eos_length = 1
    #     if self.tokenizer.eos_token is not None:
    #         tokens = tokens + [self.tokenizer.eos_token]
    #     elif self.tokenizer.sep_token is not None:
    #         tokens = tokens + [self.tokenizer.sep_token]
    #     else:
    #         eos_length = 0
    #
    #     indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
    #
    #     # https://github.com/huggingface/transformers/pull/2509
    #     # once it's fixed, you can get rid of this
    #     if self.model_name == 'xlm-roberta-large':
    #         indexed_tokens[id_to_mask + bos_length] = 250001
    #
    #     if self.model_predict is None:
    #         self.model_predict = self.__model_predict.from_pretrained(self.model_name, cache_dir=CACHE_DIR).to(self.device)
    #         self.model_predict.eval()
    #     indexed_tokens = torch.tensor([indexed_tokens]).to(self.device)
    #     with torch.no_grad():
    #         predictions = self.model_predict(indexed_tokens)[0][0][eos_length:-bos_length]
    #         prob_dist = torch.softmax(predictions, dim=-1)
    #         prob_dist_masked = prob_dist[id_to_mask]
    #         values, indices = prob_dist_masked.topk(top_k_predict, dim=-1)
    #         predicted_token_k = self.ids_to_tokens(indices.cpu().numpy())
    #
    #         # log-likelihood
    #         likeli = math.log(prob_dist_masked[target_token_id])
    #     return predicted_token_k, likeli

    # def tokens_to_ids(self, tokens):
    #     """ list of ids -> list of tokens """
    #     return self.tokenizer.convert_tokens_to_ids(tokens)
    #
    # def ids_to_tokens(self, ids):
    #     """ list of ids -> list of tokens """
    #     return self.tokenizer.convert_ids_to_tokens(ids)
    #
    # def tokens_to_string(self, tokens: list):
    #     """ list of tokens -> string """
    #     return self.tokenizer.convert_tokens_to_string(tokens)

    # def perplexity(self, tokens: list, return_ll: bool = False):
    #     """ (pseudo) perplexity
    #
    #     :param tokens: list of token
    #     :param return_ll: bool if return token-wise log likelihood
    #     :return:
    #     """
    #
    #     def get_likelihood(_id):
    #         _, log_ll = self.predict(tokens, id_to_mask=_id, top_k_predict=1)
    #         return log_ll
    #
    #     log_lls = [get_likelihood(_i) for _i in range(len(tokens))]
    #     ppl = math.exp(- sum(log_lls) / len(log_lls))
    #     if return_ll:
    #         return ppl, log_lls
    #     else:
    #         return ppl

    # def predict_ml(self,
    #                tokens: list,
    #                id_to_mask: int = None,
    #                top_k_predict: int = 10,
    #                worst_k_token: int = 3):
    #     """ conditional maximum likelihood-based prediction
    #
    #     :param tokens: list of token
    #     :param id_to_mask: single token prediction
    #     :param top_k_predict: top k sampling for prediction
    #     :param worst_k_token: worst k replacement candidate
    #     :return:
    #     """
    #     tokens = tokens.copy()
    #     original_ppl, log_likeli = self.perplexity(tokens, return_ll=True)
    #
    #     def mle_edit_single_token(__id):
    #
    #         # skip token
    #         if log_likeli[__id] is None:
    #             return tokens, original_ppl
    #
    #         tokens_c = tokens.copy()
    #         pred, likeli = self.predict(tokens, id_to_mask=__id, top_k_predict=top_k_predict)
    #
    #         def replace_token(p):
    #             __t = tokens_c.copy()
    #             __t[__id] = p
    #             return __t
    #
    #         ppls = [[p, self.perplexity(replace_token(p))] for p in pred]
    #         ppls = sorted(ppls, key=lambda x: x[1])
    #         best_ppl = ppls[0][1]
    #         if best_ppl > original_ppl:
    #             return tokens_c, original_ppl
    #         else:
    #             tokens_c[__id] = ppls[0][0]
    #             return tokens_c, best_ppl
    #
    #     if id_to_mask:
    #         # single estimation
    #         return mle_edit_single_token(id_to_mask)
    #     else:
    #         # run estimation over all token and pick the best edition
    #         if worst_k_token:  # pick worst k token as replace candidate to keep memory usage low
    #             candidate = sorted(enumerate(log_likeli), key=lambda x: (x[1] is None, x[1]))[:worst_k_token]
    #             token_ppl = [mle_edit_single_token(_i) for _i, _ in candidate]
    #         else:
    #             token_ppl = [mle_edit_single_token(_i) for _i in range(len(tokens))]
    #         token_ppl = sorted(token_ppl, key=lambda x: x[1])
    #         _best_token = token_ppl[0][0]
    #         _best_ppl = token_ppl[0][1]
    #         if _best_ppl > original_ppl:
    #             return tokens, original_ppl
    #         else:
    #             return _best_token, _best_ppl
    #

# if __name__ == '__main__':
    # example_sentence = "COVID-19 case numbers are rising rapidly across the whole of the UK and in other countries." \
    #                    "We must act now to control the spread of the virus. The single most important action we can" \
    #                    "all take, in fighting coronavirus, is to stay at home, to protect the NHS and save lives."
    # mask_id = 0
    # lm = TransformersLM('roberta-base')
    #
    # print("sample sentence: {}".format(example_sentence))
    # ex_tokens = lm.tokenize(example_sentence)
    # print("tokens: {}".format(ex_tokens))
    # print()
    # tmp_ppl = lm.perplexity(tmp)
    # print('*** perplexity ***')
    # print('- string    :', _test)
    # print('- perplexity:', tmp_ppl)
    # print('\n*** editing test ***')
    # __pred_tokens, __ppl = lm.predict_ml(tmp.copy(),
    #                                      top_k_predict=5,
    #                                      worst_k_token=3,
    #                                      skip_kanji=True,
    #                                      skip_roman=True)
    # __str = lm.tokens_to_string(__pred_tokens)
    # print('- output    :', __str)
    # print('- perplexity:', __ppl)
