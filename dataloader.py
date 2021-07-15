import sys; sys.path.append('../common');
from helper import *
from transformers import BertTokenizer, BertModel, BertForMaskedLM,BertForNextSentencePrediction, BertForSequenceClassification

# Main Dataloader for seq_tagger.py
class MainDataset(Dataset):
	def __init__(self, dataset, feat2dim, params):
		self.dataset	= dataset
		self.feat2dim 	= feat2dim
		self.p 		= params

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		ele	= self.dataset[idx]

		return torch.FloatTensor(ele['embed']), ele['labels'], ele

	def pad_data(self, convs):
		embed_dim 	= convs[0][0].shape[1]
		max_len   	= np.max([x[0].shape[0] for x in convs])

		# Text sequence
		conv_pad	= np.zeros((len(convs), max_len, embed_dim), np.float32)
		conv_mask	= np.zeros((len(convs), max_len), np.float32)
		conv_len	= np.int32([x[0].shape[0] for x in convs])

		# Features
		if 'semantic' in self.feat2dim:
			max_sem_type 	= np.max([np.max([len(y) for y in x[2]['semantic']]) for x in convs])
		feats = {
			'speaker'	: np.zeros((len(convs), max_len), np.int32) if 'speaker'  in self.feat2dim else [],
			'position'	: np.zeros((len(convs), max_len), np.int32) if 'position' in self.feat2dim else [],
			'semantic'	: np.zeros((2, len(convs), max_len, max_sem_type), np.int32) if 'semantic' in self.feat2dim else [],
		}

		# Labels
		labels	= {
			'med_tag'	: np.zeros((len(convs), max_len), np.int32),
			'med_class'	: np.zeros((len(convs), max_len), np.int32)
		}

		for i, (embed, label, othr) in enumerate(convs):
			conv_pad	[i,	:embed.shape[0]]	= embed
			conv_mask	[i,	:embed.shape[0]]	= 1.0
			
			if 'speaker'  		in self.feat2dim: feats['speaker'] [i, :embed.shape[0]] = othr['speaker']
			if 'position' 		in self.feat2dim: feats['position'][i, :embed.shape[0]] = othr['position']

			if 'semantic' in self.feat2dim:
				for j in range(embed.shape[0]):
					feats['semantic'][0][i, j, :len(othr['semantic'][j])] = othr['semantic'][j]
					feats['semantic'][1][i, j, :len(othr['semantic'][j])] = 1
			
			labels['med_tag']  [i, :embed.shape[0]] = label['med_tag']
			labels['med_class'][i, :embed.shape[0]] = label['med_class']

		labels = {k: torch.LongTensor(v) for k, v in labels.items()}
		feats  = {k: torch.LongTensor(v) for k, v in feats.items()}

		return torch.FloatTensor(conv_pad), torch.LongTensor(conv_len), torch.FloatTensor(conv_mask), feats, labels

	def collate_fn(self, all_data):
		all_data.sort(key = lambda x: -x[0].shape[0])

		batches = []
		num_batches = int(np.ceil(len(all_data) / self.p.batch_size))

		for i in range(num_batches):
			start_idx  = i * self.p.batch_size
			data 	   = all_data[start_idx : start_idx + self.p.batch_size]
			conv_pad, conv_len, conv_mask, feats, labels = self.pad_data(data)

			batches.append ({
				'conv_pad'	: conv_pad,
				'conv_mask'	: conv_mask,
				'conv_len'	: conv_len,
				'labels'	: labels,
				'feats'		: feats,
				'_rest'		: [x[-1] for x in data],
			})

		return batches

# Main Dataset for Multi Label Setting
class MainDatasetMultiLabel(MainDataset):
	def __init__(self, dataset, feat2dim, num_class, params):
		super(MainDatasetMultiLabel, self).__init__(dataset, feat2dim, params)
		self.num_class = num_class

	def pad_data(self, convs):
		embed_dim  = convs[0][0].shape[1]
		max_len    = np.max([x[0].shape[0] for x in convs])

		# Text sequence
		conv_pad	= np.zeros((len(convs), max_len, embed_dim), np.float32)
		conv_mask	= np.zeros((len(convs), max_len), np.float32)
		conv_len	= np.int32([x[0].shape[0] for x in convs])

		# Features
		if 'semantic' in self.feat2dim:
			max_sem_type 	= np.max([np.max([len(y) for y in x[2]['semantic']]) for x in convs])
		feats = {
			'speaker'		: np.zeros((len(convs), max_len), np.int32) if 'speaker'  in self.feat2dim else [],
			'position'		: np.zeros((len(convs), max_len), np.int32) if 'position' in self.feat2dim else [],
			'semantic'		: np.zeros((2, len(convs), max_len, max_sem_type), np.int32) if 'semantic' in self.feat2dim else []
		}

		# Labels
		labels	= {
			'med_tag'	: np.zeros((len(convs), max_len), np.int32),
			'med_class'	: np.zeros((len(convs), max_len, self.num_class['med_class']), np.int32)
		}



		for i, (embed, label, othr) in enumerate(convs):
			conv_pad	[i,	:embed.shape[0]]	= embed
			conv_mask	[i,	:embed.shape[0]]	= 1.0
			
			if 'speaker'  		in self.feat2dim: feats['speaker'] [i, :embed.shape[0]] = othr['speaker']
			if 'position' 		in self.feat2dim: feats['position'][i, :embed.shape[0]] = othr['position']

			if 'semantic' in self.feat2dim:
				for j in range(embed.shape[0]):
					feats['semantic'][0][i, j, :len(othr['semantic'][j])] = othr['semantic'][j]
					feats['semantic'][1][i, j, :len(othr['semantic'][j])] = 1
			
			labels['med_tag']  [i, :embed.shape[0]] = label['med_tag']
			labels['med_class'][i, :embed.shape[0]] = label['med_class']

		labels = {k: torch.LongTensor(v) for k, v in labels.items()}
		for k, v in feats.items():
			if "logs" not in k:
				feats[k] = torch.LongTensor(v)
			else:
				feats[k] = torch.FloatTensor(v)

		return torch.FloatTensor(conv_pad), torch.LongTensor(conv_len), torch.FloatTensor(conv_mask), feats, labels

	def collate_fn(self, all_data):
		all_data.sort(key = lambda x: -x[0].shape[0])

		batches = []
		num_batches = int(np.ceil(len(all_data) / self.p.batch_size))

		for i in range(num_batches):
			start_idx  = i * self.p.batch_size
			data 	   = all_data[start_idx : start_idx + self.p.batch_size]
			conv_pad, conv_len, conv_mask, feats, labels = self.pad_data(data)

			batches.append ({
				'conv_pad'	: conv_pad,
				'conv_mask'	: conv_mask,
				'conv_len'	: conv_len,
				'labels'	: labels,
				'feats'		: feats,
				'_rest'		: [x[-1] for x in data],
			})

		return batches

class BertDatasetMultiLabel(MainDataset):
	def __init__(self, dataset, feat2dim, num_class, params, tokenizer):
		super(BertDatasetMultiLabel, self).__init__(dataset, feat2dim, params)
		self.num_class	= num_class
		self.tokenizer	= tokenizer

	def __getitem__(self, idx):
		ele				= self.dataset[idx]
		conv_tokens 	= [["[CLS]"] + self.tokenizer.tokenize(Xi['txt'])[:self.p.max_utt_len - 2] + ["[SEP]"] for Xi in ele['transcript']]

		conv_token_ids	= [self.tokenizer.convert_tokens_to_ids(utt_tokens) + [0]*(self.p.max_utt_len - len(utt_tokens)) for utt_tokens in conv_tokens]
		conv_seg_ids	= [[0]*len(utt_tokens) + [0]*(self.p.max_utt_len - len(utt_tokens)) for utt_tokens in conv_tokens]
		conv_att_mask	= [[1]*len(utt_tokens) + [0]*(self.p.max_utt_len - len(utt_tokens)) for utt_tokens in conv_tokens]

		return (conv_tokens, np.array(conv_token_ids)), np.array(conv_seg_ids), np.array(conv_att_mask), ele['labels'], ele

	def pad_data(self, convs):
		max_len    = np.max([x[0][1].shape[0] for x in convs])

		# Text sequence
		conv_b_token_ids= np.zeros((len(convs), max_len, self.p.max_utt_len), np.float32)
		conv_mask		= np.zeros((len(convs), max_len), np.float32)
		conv_b_seg_ids	= np.zeros((len(convs), max_len, self.p.max_utt_len), np.float32)
		conv_b_att_mask	= np.zeros((len(convs), max_len, self.p.max_utt_len), np.float32)
		conv_len		= np.int32([x[0][1].shape[0] for x in convs])

		# Features
		if 'semantic' in self.feat2dim:
			max_sem_type 	= np.max([np.max([len(y) for y in x[2]['semantic']]) for x in convs])
		feats = {
			'speaker'	: np.zeros((len(convs), max_len), np.int32) if 'speaker'  in self.feat2dim else [],
			'position'	: np.zeros((len(convs), max_len), np.int32) if 'position' in self.feat2dim else [],
			'semantic'	: np.zeros((2, len(convs), max_len, max_sem_type), np.int32) if 'semantic' in self.feat2dim else [] # First dim stores indices and  mask respectively
		}

		# Labels
		labels	= {
			'med_tag'	: np.zeros((len(convs), max_len), np.int32),
			'med_class'	: np.zeros((len(convs), max_len, self.num_class['med_class']), np.int32),
			'sym_label'	: np.zeros((len(convs), max_len, self.num_class['sym_label']), np.int32),
			'chief_comp'	: np.zeros((len(convs)), np.int32)
		}

		for i, (tokens, conv_seg_ids, conv_att_mask, label, othr) in enumerate(convs):
			if tokens[1].shape == (208,):
				import pdb; pdb.set_trace()
			conv_b_token_ids	[i,	:tokens[1].shape[0],:]	= tokens[1]
			conv_mask			[i,	:tokens[1].shape[0]]	= 1.0
			conv_b_seg_ids		[i,	:tokens[1].shape[0],:]	= conv_seg_ids
			conv_b_att_mask		[i,	:tokens[1].shape[0],:]	= conv_att_mask
			
			if 'speaker'  in self.feat2dim: feats['speaker'] [i, :tokens[1].shape[0]] = othr['speaker']
			if 'position' in self.feat2dim: feats['position'][i, :tokens[1].shape[0]] = othr['position']

			if 'semantic' in self.feat2dim:
				for j in range(tokens[1].shape[0]):
					feats['semantic'][0][i, j, :len(othr['semantic'][j])] = othr['semantic'][j]
					feats['semantic'][1][i, j, :len(othr['semantic'][j])] = 1
			
			labels['med_tag']  [i, :tokens[1].shape[0]] = label['med_tag']
			labels['med_class'][i, :tokens[1].shape[0]] = label['med_class']
			labels['sym_label'][i, :tokens[1].shape[0]] = label['sym_label']
			labels['chief_comp'][i] 					= label['chief_comp']

		labels = {k: torch.LongTensor(v) for k, v in labels.items()}
		for k, v in feats.items():
			if "logs" not in k:
				feats[k] = torch.LongTensor(v)
			else:
				feats[k] = torch.FloatTensor(v)

		return 	torch.LongTensor(conv_b_token_ids), torch.LongTensor(conv_mask),	\
				torch.LongTensor(conv_b_seg_ids), torch.LongTensor(conv_b_att_mask),	\
				torch.LongTensor(conv_len), feats, labels

	def collate_fn(self, all_data):
		all_data.sort(key = lambda x: -len(x[0][1]))

		batches = []
		num_batches = int(np.ceil(len(all_data) / self.p.batch_size))

		for i in range(num_batches):
			start_idx  = i * self.p.batch_size
			data 	   = all_data[start_idx : start_idx + self.p.batch_size]
			conv_b_token_ids, conv_mask, conv_b_seg_ids, conv_b_att_mask, conv_len, feats, labels = self.pad_data(data)

			batches.append ({
				'conv_b_token_ids'	: conv_b_token_ids,
				'conv_b_att_mask'	: conv_b_att_mask,
				'conv_b_seg_ids'	: conv_b_seg_ids,
				'conv_mask'	: conv_mask,
				'conv_len'	: conv_len,
				'labels'	: labels,
				'feats'		: feats,
				'_rest'		: [x[-1] for x in data],
			})

		return batches


class BertFineDataset(Dataset):
	def __init__(self, dataset, num_class, params):
		self.dataset	= dataset
		self.num_class 	= num_class
		self.p 		= params

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		ele = self.dataset[idx]
		return ele['text'], ele['labels']

	def pad_data(self, convs):
		max_utter 	= np.max([len(conv[0]) for conv in convs])
		max_seq_len 	= np.max([np.max([len(x) for x in conv[0]]) for conv in convs])

		tok_pad		= np.zeros((len(convs), max_utter, max_seq_len), np.int32)
		tok_mask	= np.zeros((len(convs), max_utter, max_seq_len), np.float32)
		utter_len 	= np.zeros((len(convs), max_utter), np.int32)

		labels	= {
			'med_class': np.zeros((len(convs), max_utter, self.num_class['med_class']), np.int32),
		}

		for i, (text, label) in enumerate(convs):
			for j in range(len(text)):
				tok_pad [i][j, :len(text[j])] = text[j]
				tok_mask[i][j, :len(text[j])] = 1.0
				if self.p.mask_unk and (103 in text[j]):
					tok_mask[i][j, text[j].index(103)] = 0.0
				utter_len[i][j]		      = len(text[j])

				labels['med_class'][i][j] = label['med_class'][j]

		labels = {k: torch.LongTensor(v) for k, v in labels.items()}

		return torch.LongTensor(tok_pad), torch.FloatTensor(tok_mask), torch.LongTensor(utter_len), labels

	def collate_fn(self, all_data):
		all_data.sort(key = lambda x: -len(x[0]))

		batches = []
		num_batches = int(np.ceil(len(all_data) / self.p.batch_size))

		for i in range(num_batches):
			start_idx  = i * self.p.batch_size
			data 	   = all_data[start_idx : start_idx + self.p.batch_size]

			tok_pad, tok_mask, utter_len, labels = self.pad_data(data)

			batches.append ({
				'tok_pad'	: tok_pad,
				'tok_mask'	: tok_mask,
				'utter_len'	: utter_len,
				'labels'	: labels,
			})

		return batches


class BertDataset(Dataset):
	def __init__(self, dataset, params):
		self.dataset	= dataset
		self.p 		= params

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		ele = self.dataset[idx]
		return ele['text'], torch.LongTensor(ele['med_class']), ele['transcript']

	def pad_data(self, convs):
		max_utter 	= np.max([len(x[0]) for x in convs])
		max_len		= np.max([np.max([len(y) for y in x[0]]) for x in convs])

		tok_pad		= np.zeros((len(convs), max_utter, max_len), np.int32)
		tok_mask	= np.zeros((len(convs), max_utter, max_len), np.float32)
		conv_mask	= np.zeros((len(convs), max_utter), np.float32)

		label_pad	= np.zeros((len(convs), max_utter), np.int32)
		conv_len	= np.int32([len(x[0]) for x in convs])

		for i, (text, tar, _) in enumerate(convs):
			for j, utter in enumerate(text):
				tok_pad [i, j, :len(utter)] = utter
				tok_mask[i, j, :len(utter)] = 1.0

			conv_mask[i, :len(text)] = 1.0
			label_pad[i, :len(text)] = tar

		return torch.LongTensor(tok_pad), torch.LongTensor(conv_len), torch.FloatTensor(tok_mask), torch.FloatTensor(conv_mask), torch.LongTensor(label_pad)

	def collate_fn(self, all_data):
		all_data.sort(key = lambda x: len(x[0]))

		batches = []
		num_batches = int(np.ceil(len(all_data) / self.p.batch_size))

		for i in range(num_batches):
			start_idx  = i * self.p.batch_size
			data 	   = all_data[start_idx : start_idx + self.p.batch_size]

			tok_pad, conv_len, tok_mask, conv_mask, label_pad = self.pad_data(data)

			batches.append ({
				'tok_pad'	: tok_pad,
				'tok_mask'	: tok_mask,
				'conv_mask'	: conv_mask,
				'conv_len'	: conv_len,
				'labels'	: label_pad,
				'_transcript'	: [x[-1] for x in data]
			})

		return batches


class BertReorgDataset(Dataset):
	def __init__(self, dataset, num_class, params):
		self.dataset	= dataset
		self.num_class 	= num_class
		self.p 		= params

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		ele = self.dataset[idx]
		return ele['text'], ele['labels']

	def pad_data(self, utters):
		max_len  = min(np.max([len(x[0]) for x in utters]), self.p.max_seq_len)
		tok_pad	 = np.zeros((len(utters), max_len), np.int32)
		tok_mask = np.zeros((len(utters), max_len), np.float32)

		labels	= {
			'med_class': np.zeros((len(utters), self.num_class['med_class']), np.int32),
		}

		for i, (text, label) in enumerate(utters):
			tok_pad [i, :len(text)] = text[:max_len]
			tok_mask[i, :len(text)] = 1.0

			labels['med_class'][i] = label['med_class']

		labels = {k: torch.LongTensor(v) for k, v in labels.items()}

		return torch.LongTensor(tok_pad), torch.FloatTensor(tok_mask), labels

	def collate_fn(self, all_data):
		all_data.sort(key = lambda x: -len(x[0]))

		batches = []
		num_batches = int(np.ceil(len(all_data) / self.p.batch_size))

		for i in range(num_batches):
			start_idx  = i * self.p.batch_size
			data 	   = all_data[start_idx : start_idx + self.p.batch_size]

			tok_pad, tok_mask, labels = self.pad_data(data)

			batches.append ({
				'tok_pad'	: tok_pad,
				'tok_mask'	: tok_mask,
				'labels'	: labels,
			})

		return batches