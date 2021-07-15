import os, sys, pdb, numpy as np, random, argparse, codecs, pickle, time, json, csv, copy
import logging, logging.config, itertools, pathlib, socket, gspread, warnings

from nltk.tokenize import word_tokenize
from tqdm import tqdm
from pprint import pprint
from pymongo import MongoClient
from collections import OrderedDict
from sklearn.exceptions import UndefinedMetricWarning
from joblib import Parallel, delayed
from collections import defaultdict as ddict, Counter
from sklearn.metrics import f1_score, classification_report, roc_auc_score
from oauth2client.service_account import ServiceAccountCredentials

# Pytorch related imports
import torch, torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn import Parameter as Param
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import BertTokenizer, BertModel, BertConfig, BertPreTrainedModel
from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup

def makeDirectory(dirpath):
	if not os.path.exists(dirpath):
		os.makedirs(dirpath)

def checkFile(filename):
	return pathlib.Path(filename).is_file()

def str_proc(x):
	return str(x).strip().lower()

def set_gpu(gpus):
	os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = gpus

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)
	
def partition(lst, n):
        division = len(lst) / float(n)
        return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]

def getChunks(inp_list, chunk_size):
	return [inp_list[x:x+chunk_size] for x in range(0, len(inp_list), chunk_size)]

def mergeList(list_of_list):
	return list(itertools.chain.from_iterable(list_of_list))

def mergeListInDictOne(list_of_dict_of_list):

	dict_of_lists = {key: [] for key, values in list_of_dict_of_list.items()}

	for key, values in list_of_dict_of_list.items():
		for val in values:
			dict_of_lists[key].extend(val)

	return dict_of_lists

def mergeListInDict(list_of_dict_of_list):

	dict_of_lists = {key: [] for key, values in list_of_dict_of_list[0].items()}

	for itm in list_of_dict_of_list:
		for key, values in itm.items():
			dict_of_lists[key].append(values)

	return dict_of_lists


def get_logger(name, log_dir, config_dir):
	config_dict = json.load(open('{}/log_config.json'.format(config_dir)))
	config_dict['handlers']['file_handler']['filename'] = '{}/{}'.format(log_dir, name.replace('/', '-'))
	logging.config.dictConfig(config_dict)
	logger = logging.getLogger(name)

	std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
	consoleHandler = logging.StreamHandler(sys.stdout)
	consoleHandler.setFormatter(logging.Formatter(std_out_format))
	logger.addHandler(consoleHandler)

	return logger

def to_gpu(batch, dev):
	batch_gpu = {}
	for key, val in batch.items():
		if   key.startswith('_'):		batch_gpu[key] = val
		elif type(val) == type({1:1}): 	batch_gpu[key] = {k: v.to(dev) for k, v in batch[key].items()}
		else: 				batch_gpu[key] = val.to(dev)
	return batch_gpu

def read_csv(fname):
	with open(fname) as f:
		f.readline()
		for data in csv.reader(f):
			yield data

def mean_dict(acc):
	return {k: np.round(np.mean(v), 3) for k, v in acc.items()}

def get_param(shape):
	param = Parameter(torch.Tensor(*shape))	
	xavier_normal_(param.data)
	return param

def comb_dict(res):
	return {k: [x[k] for x in res] for k in res[0].keys()}


def getGlove(trans_list, dim, db_glove):

	trans_tok = [word_tokenize(trans['txt']) for trans in trans_list]
	wrd_list  = list(set(mergeList(trans_tok)))

	embed_map = {}
	res = db_glove.find({"_id": {"$in": wrd_list}})
	for ele in res:
		embed_map[ele['_id']] = np.float32(ele['vec'])

	embeds	  = np.zeros((len(trans_list), dim), np.float32)

	for i, trans in enumerate(trans_tok):
		embed_list = [embed_map[wrd] for wrd in trans if wrd in embed_map]
		if len(embed_list) == 0: embeds[i, :] = np.random.randn(dim)
		else:			 embeds[i, :] = np.mean(embed_list)
		
	return embeds


def getContext(data):
	for conv in data:
		for_context = [[-1,-1, -1] for u in range(len(conv['transcript']))]
		prev_context = [[-1,-1, -1] for u in range(len(conv['transcript']))]
		
		for u, utt in conv['transcript'].items():
	#         print(utt)
			for u1 in range(int(u)+1, len(conv['transcript'])):
				if all(v > -1 for v in for_context[int(u)]):
					break
				if for_context[int(u)][0] == -1 and conv['transcript'][str(u1)]['speaker'] == 'PT':
	#                 print("WTF")
					for_context[int(u)][0] = u1
				elif for_context[int(u)][1] == -1 and conv['transcript'][str(u1)]['speaker'] == 'DR':
					for_context[int(u)][1] = u1
				elif for_context[int(u)][2] == -1 and conv['transcript'][str(u1)]['speaker'] != 'DR' and conv['transcript'][str(u1)]['speaker'] != 'PT':
					for_context[int(u)][2] = u1
					
		for u in range(len(conv['transcript'])-1, -1, -1):
	#         print(utt)
			for u1 in range(u-1, -1, -1):
				if all(v > -1 for v in prev_context[int(u)]):
					break
				if prev_context[int(u)][0] == -1 and conv['transcript'][str(u1)]['speaker'] == 'PT':
	#                 print("WTF")
					prev_context[int(u)][0] = u1
				elif prev_context[int(u)][1] == -1 and conv['transcript'][str(u1)]['speaker'] == 'DR':
					prev_context[int(u)][1] = u1
				elif prev_context[int(u)][2] == -1 and conv['transcript'][str(u1)]['speaker'] != 'DR' and conv['transcript'][str(u1)]['speaker'] != 'PT':
					prev_context[int(u)][2] = u1
					
		conv['for_context'] = for_context
		conv['prev_context'] = prev_context

	return data