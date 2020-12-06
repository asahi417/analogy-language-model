# """ pre-trained LM for sentence evaluation """
# import os
# import logging
# import math
# from typing import List
# from itertools import chain
# logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
#
# import torch
# import transformers
# from tqdm import tqdm
#
#
# os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning message
#
#
# __all__ = 'TransformersLM'
#
#
# class Dataset(torch.utils.data.Dataset):
#     """ `torch.utils.data.Dataset` """
#     float_tensors = ['attention_mask']
#
#     def __init__(self, data: List):
#         self.data = data  # a list of dictionaries
#
#     def __len__(self):
#         return len(self.data)
#
#     def to_tensor(self, name, data):
#         if name in self.float_tensors:
#             return torch.tensor(data, dtype=torch.float32)
#         return torch.tensor(data, dtype=torch.long)
#
#     def __getitem__(self, idx):
#         return {k: self.to_tensor(k, v) for k, v in self.data[idx].items()}
#
#
# class TransformersLM:
#     """ transformers language model based sentence-mining """
#
#     def __init__(self,
#                  model: str,
#                  max_length: int = None,
#                  cache_dir: str = './cache',
#                  num_worker: int = 1):
#         """ transformers language model based sentence-mining
#
#         :param model: a model name corresponding to a model card in `transformers`
#         :param max_length: a model max length if specified, else use model_max_length
#         """
#         logging.info('*** setting up a language model ***')
#         self.num_worker = num_worker
#         # model setup
#         self.tokenizer = transformers.AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)
#         self.config = transformers.AutoConfig.from_pretrained(model, cache_dir=cache_dir)
#         self.model = transformers.AutoModelForMaskedLM.from_pretrained(model, config=self.config, cache_dir=cache_dir)
#         self.model.eval()
#         if max_length:
#             assert self.tokenizer.model_max_length >= max_length
#             self.max_length = max_length
#         else:
#             self.max_length = self.tokenizer.model_max_length
#         # gpu
#         self.n_gpu = torch.cuda.device_count()
#         assert self.n_gpu <= 1
#         self.device = 'cuda' if self.n_gpu > 0 else 'cpu'
#         self.model.to(self.device)
#         logging.info('running on %i GPU' % self.n_gpu)
#         # sentence prefix tokens
#         tokens = self.tokenizer.tokenize('get tokenizer specific prefix')
#         tokens_encode = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode('get tokenizer specific prefix'))
#         self.sp_token_prefix = tokens_encode[:tokens_encode.index(tokens[0])]
#         self.sp_token_suffix = tokens_encode[tokens_encode.index(tokens[-1]) + 1:]
#
#     def batch_encode_plus_mask(self,
#                                texts: List,
#                                head_token: List,
#                                head_token: List,
#                                batch_size: int = None):
#         """ to get batch data_loader with `self.encode_plus_masked` function
#
#         :param texts: a list of texts
#         :param head_token: a list of string tokens to be masked
#         :param batch_size:
#         :return: `torch.utils.data.DataLoader` class
#         """
#         batch_size = len(texts) if batch_size is None else batch_size
#         assert len(texts) == len(head_token), "size mismatch: {} vs {}".format(len(texts), len(head_token))
#         data = list(map(lambda x: self.encode_plus_mask(*x), zip(texts, head_token)))
#         if self.num_worker == 1:
#             os.environ["OMP_NUM_THREADS"] = "1"  # to turn off warning message
#         data_loader = torch.utils.data.DataLoader(
#             Dataset(data), num_workers=self.num_worker, batch_size=batch_size, shuffle=False, drop_last=False)
#         return data_loader
#
#     def encode_plus_mask(self,
#                          text: str,
#                          token_to_mask: str,
#                          padding: bool = True,
#                          longest_subword_search: bool = True):
#         """ to get an output from `encode_plus` with a masked token specified by a string
#         Note: it can only take single token, and a phrase over multiple tokens will raise error
#
#         :param str text: a text to encode
#         :param str token_to_mask: a target token to be masked
#         :param bool padding: a flag to encode output with/without padding
#         :param bool longest_subword_search: a flag to enable subword serach if token_to_mask is not in the tokens
#         :return encode
#         """
#         assert len(text.replace(' ', '')) != 0, 'found an empty text'
#
#         token_list = self.tokenizer.tokenize(text)
#         assert len(token_list) <= self.max_length,\
#             "a token size exceeds the max_length: {} > {}".format(len(token_list), self.max_length)
#
#         # get first token in the list which is exact `mask_string`
#         # mask_positions = [n for n, t in enumerate(token_list) if token_to_mask in t]
#         mask_positions = list(filter(lambda x: token_to_mask in x[1], enumerate(token_list)))
#         if len(mask_positions) > 0:
#             mask_position = mask_positions[0][0]
#         elif not longest_subword_search:
#             raise ValueError('`{}` is not found in tokens `{}`'.format(token_to_mask, token_list))
#         else:
#             # search by the shortest subword that has overlap with the original token to be masked
#             subword = sorted(filter(lambda x: x in token_to_mask, token_list), key=lambda x: len(x), reverse=True)[0]
#             mask_position = token_list.index(subword)
#
#         # mask the token and keep the mask position, masked token, masked token id
#         # * note that `the<mask> is ~` == `the <mask> is ~` in the tokenizer module
#         mask_token_id = self.tokenizer.convert_tokens_to_ids(token_list[mask_position])
#         token_list[mask_position] = self.tokenizer.mask_token
#         mask_position += len(self.sp_token_prefix)  # shift for prefix
#
#         # encode sentence into model input format as a batch with single data
#         encode = self.tokenizer.encode_plus(
#             token_list, max_length=self.max_length, padding='max_length' if padding else False, truncation=padding)
#         encode['mask_position'] = mask_position
#         encode['mask_token_id'] = mask_token_id
#         return encode
#
#     def get_nll(self,
#                 texts: (List, str),
#                 head_token: (List, str),
#                 batch_size: int = None,
#                 top_k_predict: int = 10):
#         """ get negative log likelihood of a masked token within a sentence
#
#         :param texts:
#         :param head_token:
#         :param batch_size:
#         :param top_k_predict:
#         :return: negative_log_likelihood, (topk_prediction_values, topk_prediction_indices)
#             log_likelihood, a list of negative log likelihood, (len(texts))
#             topk_prediction_indices, top k tokens predicted for the masked position, (len(texts)), top_k)
#             topk_prediction_values, probability along with the prediction, (len(texts)), top_k)
#         """
#         assert type(texts) == type(head_token), '`texts` and `head_token` should be same type'
#
#         if type(texts) is list and type(head_token) is list:
#             data_loader = self.batch_encode_plus_mask(texts=texts, head_token=head_token, batch_size=batch_size)
#         else:
#             encode = self.encode_plus_mask(text=texts, token_to_mask=head_token, padding=False)
#             data_loader = [{k: torch.tensor([v]) for k, v in encode.items()}]
#
#         log_likelihood, topk_prediction_indices, topk_prediction_values \
#             = self.__get_loglikelihood_with_data_loader(data_loader, top_k_predict=top_k_predict)
#         topk_prediction_indices = list(map(
#             lambda x: list(map(lambda y: self.tokenizer.decode(y), x)),
#             topk_prediction_indices
#         ))
#         negative_log_likelihood = list(map(lambda x: -1 * x, log_likelihood))
#         return negative_log_likelihood, (topk_prediction_values, topk_prediction_indices)
#
#     def batch_encode_plus_token_wise_mask(self, texts: List, batch_size: int = None):
#         """ to get batch data_loader with `self.encode_plus_token_wise_mask` function
#
#         :param texts: a list of texts
#         :param batch_size:
#         :return: `torch.utils.data.DataLoader` class, partition (partition for each text)
#         """
#         batch_size = len(texts) if batch_size is None else batch_size
#         data = list(map(lambda x: self.encode_plus_token_wise_mask(x, padding=True), texts))
#         length = list(map(lambda x: len(x), data))
#         partition = list(map(lambda x: [sum(length[:x]), sum(length[:x+1])], range(len(length))))
#         flatten_data = list(chain(*data))
#         data_loader = torch.utils.data.DataLoader(
#             Dataset(flatten_data), num_workers=self.num_worker, batch_size=batch_size, shuffle=False, drop_last=False)
#         return data_loader, partition
#
#     def encode_plus_token_wise_mask(self, text: str, padding: bool = True):
#         """ to get a list of outputs from `encode_plus`, where each corresponds to one with a text with
#         a mask at i ~ [0, n] (n: a size of tokens in the given text)
#
#         :param str text: a text to encode
#         :param bool padding: a flag to encode output with/without paddings
#         :return encodes_list
#         """
#         assert len(text.replace(' ', '')) != 0, 'found an empty text'
#         token_list = self.tokenizer.tokenize(text)
#         assert len(token_list) <= self.max_length,\
#             "a token size exceeds the max_length: {} > {}".format(len(token_list), self.max_length)
#
#         # get encoded feature for each token masked
#         def encode_with_single_mask_id(mask_position: int):
#             _token_list = token_list.copy()  # can not be encode outputs because of prefix
#             mask_token_id = self.tokenizer.convert_tokens_to_ids(_token_list[mask_position])
#             _token_list[mask_position] = self.tokenizer.mask_token
#             encode = self.tokenizer.encode_plus(
#                 _token_list, max_length=self.max_length, padding='max_length' if padding else False, truncation=padding)
#             encode['mask_position'] = mask_position + len(self.sp_token_prefix)
#             encode['mask_token_id'] = mask_token_id
#             return encode
#
#         encodes_list = [encode_with_single_mask_id(i) for i in range(len(token_list))]
#         return encodes_list
#
#     def get_pseudo_perplexity(self, texts: (List, str), batch_size: int = None):
#         """ to compute a pseudo perplexity (mask each token and use log likelihood for each prediction for the mask)
#
#         :param texts:
#         :param batch_size:
#         :return:
#         """
#         if type(texts) is list:
#             data_loader, partition = self.batch_encode_plus_token_wise_mask(texts, batch_size=batch_size)
#         else:
#             encode_list = self.encode_plus_token_wise_mask(texts)
#             data_loader = [{k: torch.tensor([v]) for k, v in encode.items()} for encode in encode_list]
#             partition = [[0, len(data_loader)]]
#
#         list_ppl = []
#         loglikelihood, _, _ = self.__get_loglikelihood_with_data_loader(data_loader)
#         for start, end in partition:
#             sentence_loglikeli = loglikelihood[start:end]
#             list_ppl += [math.exp(- sum(sentence_loglikeli) / len(sentence_loglikeli))]
#
#         return list_ppl
#
#     def get_pmi(self,
#                 texts: (List, str),
#                 head_tokens: (List, str),
#                 tail_tokens: (List, str),
#                 batch_size: int = None):
#         """ to compute a point-wise mutual information
#
#         :param texts:
#         :param batch_size:
#         :return:
#         """
#         assert type(texts) == type(head_tokens) == type(tail_tokens), '`texts` and `head/tail` should be same type'
#         if type(texts) is list and type(head_tokens) is list:
#             data_loader = self.batch_encode_plus_mask(texts=texts, target_tokens=target_tokens, batch_size=batch_size)
#         else:
#             encode = self.encode_plus_mask(text=texts, token_to_mask=target_tokens, padding=False)
#             data_loader = [{k: torch.tensor([v]) for k, v in encode.items()}]
#
#         log_likelihood, topk_prediction_indices, topk_prediction_values \
#             = self.__get_loglikelihood_with_data_loader(data_loader, top_k_predict=top_k_predict)
#         topk_prediction_indices = list(map(
#             lambda x: list(map(lambda y: self.tokenizer.decode(y), x)),
#             topk_prediction_indices
#         ))
#         negative_log_likelihood = list(map(lambda x: -1 * x, log_likelihood))
#         return negative_log_likelihood, (topk_prediction_values, topk_prediction_indices)
#
#
#     def __get_loglikelihood_with_data_loader(self, data_loader, top_k_predict: int = None):
#         """ to run a prediction on a masked token with MLM
#
#         :param data_loader:
#         :param top_k_predict:
#         :return:
#         """
#         topk_prediction_indices = []
#         topk_prediction_values = []
#         log_likelihood = []
#         with torch.no_grad():
#             for encode in tqdm(data_loader):
#                 mask_position = encode.pop('mask_position').cpu().detach().int().tolist()
#                 mask_token_id = encode.pop('mask_token_id').cpu().detach().int().tolist()
#
#                 # get probability/prediction for batch features
#                 encode = {k: v.to(self.device) for k, v in encode.items()}
#                 logit = self.model(**encode)[0]
#                 prob = torch.softmax(logit, -1)
#
#                 # compute likelihood of masked positions given the masked tokens
#                 log_likelihood += list(map(
#                     lambda x: math.log(float(prob[x[0]][x[1][0]][x[1][1]].cpu())),
#                     enumerate(zip(mask_position, mask_token_id))
#                 ))
#
#                 # top-k prediction
#                 if top_k_predict:
#                     top_k = list(map(
#                         lambda x: list(map(
#                             lambda y: y.cpu().tolist(),
#                             prob[x[0]][x[1]].topk(top_k_predict))),
#                         enumerate(mask_position)
#                     ))
#                     topk_prediction_values += list(list(zip(*top_k))[0])
#                     topk_prediction_indices += list(list(zip(*top_k))[1])
#
#         return log_likelihood, topk_prediction_indices, topk_prediction_values
