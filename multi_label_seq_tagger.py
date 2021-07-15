from models import *
import sys; sys.path.append('../common'); 
from helper import *
from dataloader import *

from sklearn.metrics import average_precision_score, roc_auc_score

torch.autograd.set_detect_anomaly(True)

class Main(object):

	def load_data(self):

		self.data   = {'train': [], 'valid': [], 'test': []}

		self.tag2id_all = {
			'symptoms'				: 0,
			'chief_complaint'		: 1,
			'medications'			: 2,
			'prescription'			: 2,
		}

		if self.p.ind_cat != "all":
			cats = self.p.ind_cat.split(",")
			self.tag2id = {cat: 0 for cat in cats}
		else:
			self.tag2id = self.tag2id_all

		

		self.num_class = {
			'med_tag'	: 2,
			'med_class'	: len(set(self.tag2id.values()))
		}
			

		self.speaker2id		= {'PT': 0, 'DR': 1, 'REST': 2}
		self.semantic2id 	= json.load(open('{}/qumls_type2id.json'.format(self.p.res_dir)))

		# self.semantic_map	= pickle.load(open('{}/features/semantic.pkl'.format(self.p.data_dir), 'rb'))
		self.umls_map		= pickle.load(open('{}/features/umls_embed.pkl'.format(self.p.data_dir), 'rb'))
		self.semantic_map	= pickle.load(open('{}/features/quickumls.pkl'.format(self.p.data_dir), 'rb'))
		
		self.feat2dim	= OrderedDict(zip(self.p.feat.split(','), zip([int(x) for x in self.p.feat_dim.split(',')], [int(x) for x in self.p.feat_cat.split(',')])))
		self.loss2fact	= OrderedDict(zip(self.p.loss.split(','), [float(x) for x in self.p.loss_fact.split(',')]))

		"""
		feat_corr = ddict(lambda: ddict(lambda: ddict(int)))
		self.id2speaker 	= {v: k for k, v in self.speaker2id.items()}
		self.id2tag 		= {v: k for k, v in self.tag2id.items()}
		self.id2semantic 	= {v: k for k, v in self.semantic2id.items()}
		self.semantic2name 	= json.load(open('{}/qumls_type2name.json'.format(self.p.res_dir)))
		"""

		for line in tqdm(open('{}/main.json'.format(self.p.data_dir))):
			conv			= json.loads(line)
			_id			= conv['meta']['id']
			_, conv['transcript']	= zip(*sorted(conv['transcript'].items(), key = lambda x: int(x[0])))
			num_utter 		= len(conv['transcript'])

			# Get pre-trained embeddings
			conv['embed'] = np.concatenate([pickle.load(open('{}/embeddings/{}/{}.pkl'.format(self.p.embed_dir, x, _id), 'rb'))['embeddings'] for x in self.p.embed.split(',')], axis=1)

			# Get Features to be used
			if 'speaker'  in self.feat2dim: conv['speaker']  = [self.speaker2id.get(x['speaker'], self.speaker2id['REST']) for x in conv['transcript']]
			if 'position' in self.feat2dim: conv['position'] = np.int32(mergeList([i+np.zeros(len(x)) for i, x in enumerate(partition(range(num_utter), self.feat2dim['position'][1]))]))
			if 'semantic' in self.feat2dim: 
				# TODO: Should experiment with converting into sets rather than lists.
				conv['semantic'] = [mergeList([list(ent[0]['semtypes']) for ent in utter_ent]) for utter_ent in self.semantic_map[_id]]
				conv['semantic'] = [[self.semantic2id[y] for y in x if y not in ['T033', 'T170', 'T109', 'T041']] for x in conv['semantic']]	# Ignoring Intellectual Property (T170) & Finding (T033) & Organic Chemical (T109)
				
			if 'umls' in self.feat2dim:
				umls_dim, umls_cat 	= self.feat2dim['umls']
				umls_embed		= np.zeros((num_utter, umls_dim * umls_cat), np.float32)

				for i in range(num_utter):
					if len(self.umls_map[_id][i]) == 0: continue
					embed = np.concatenate(self.umls_map[_id][i][:umls_cat], axis=0)
					umls_embed[i, :embed.shape[0]] = embed
					
				conv['embed'] = np.concatenate([conv['embed'], umls_embed], axis=1)

			# Get Medical Class
			med_tag		= np.zeros(num_utter)
			med_class	= np.zeros((num_utter, self.num_class['med_class']))
			for mtype, mentions in conv['spans'].items():
				if mtype not in self.tag2id: continue
				for ele in mentions:
					med_tag[np.int32(ele['span'])] = 1
					med_class[np.int32(ele['span']), self.tag2id[mtype]] = 1

			conv['labels'] = {
				'med_tag'	: med_tag,
				'med_class'	: med_class
			}

			if "bert" in self.p.model or self.p.data_split:
				sub_convs = self.get_sub_convs(conv)
				self.data[conv['split']].extend(sub_convs)
			else:
				self.data[conv['split']].append(conv)

		# Already included in conv['embed'], don't need to handle it separately
		if 'umls' in self.feat2dim: del self.feat2dim['umls']

		self.logger.info('\nDataset size -- Train: {}, Valid: {}, Test:{}'.format(len(self.data['train']), len(self.data['valid']), len(self.data['test'])))
		self.inp_dim = self.data['train'][0]['embed'].shape[1]

		def get_data_loader(split, shuffle=True):
			if "bert" not in self.p.model:
				dataset 	= MainDatasetMultiLabel(self.data[split], self.feat2dim, self.num_class, self.p)
			else:
				tokenizer	= BertTokenizer.from_pretrained(self.p.bert_model)
				dataset		= BertDatasetMultiLabel(self.data[split], self.feat2dim, self.num_class, self.p, tokenizer)
			return  DataLoader(
					dataset,
					batch_size      = self.p.batch_size * self.p.batch_factor,
					shuffle         = shuffle,
					num_workers     = max(0, self.p.num_workers),
					collate_fn      = dataset.collate_fn
				)

		self.data_iter = {
			'train'	: get_data_loader('train'),
			'valid'	: get_data_loader('valid', shuffle=False),
			'test'	: get_data_loader('test',  shuffle=False),
		}

		self.logger.info('Number of classes: {}'.format(self.num_class))

		return len(self.data['train'])

	def get_sub_convs(self, conv):
		"""
		Splits large size conversation into smaller chunks
		"""
		
		sub_convs = []
		if len(conv['transcript']) > self.p.max_conv_len:
			num_sub_convs = int(np.ceil(len(conv['transcript'])/self.p.max_conv_len))
			for i in range(num_sub_convs):
				start_ind = i*self.p.max_conv_len
				end_ind = min((i+1)*self.p.max_conv_len, len(conv['transcript']))

				sub_conv = {}
				sub_conv['transcript']	= conv['transcript'][start_ind: end_ind]
				sub_conv['meta']	= conv['meta']
				sub_conv['spans']	= conv['spans']
				sub_conv['split']	= conv['split']
				sub_conv['embed']	= conv['embed'][start_ind: end_ind]
				sub_conv['indices']	= (start_ind, end_ind, i)

				if 'speaker'  		in self.feat2dim: sub_conv['speaker']	= conv['speaker'][start_ind: end_ind]
				if 'position' 		in self.feat2dim: sub_conv['position']	= conv['position'][start_ind: end_ind]
				if 'semantic' 		in self.feat2dim: sub_conv['semantic']	= conv['semantic'][start_ind: end_ind]
				if 'bert_logs' 		in self.feat2dim: sub_conv['bert_logs']	= conv['bert_logs'][start_ind: end_ind]
				if 'bert_dia_logs' 	in self.feat2dim: sub_conv['bert_dia_logs']	= conv['bert_dia_logs'][start_ind: end_ind]
				if 'for_context'	in conv: sub_conv['for_context']		= conv['for_context'][start_ind: end_ind]
				if "prev_context"	in conv: sub_conv['prev_context']		= conv['prev_context'][start_ind: end_ind]

				sub_conv['labels'] = {
					'med_tag'	: conv['labels']['med_tag'][start_ind: end_ind],
					'med_class'	: conv['labels']['med_class'][start_ind: end_ind],
					'sym_label'	: conv['labels']['sym_label'][start_ind: end_ind],
					'chief_comp'	: conv['labels']['chief_comp']
				}
				sub_convs.append(sub_conv)

		else:
			sub_convs.append(conv)

		return sub_convs

	def add_model(self):
		if   	self.p.model.lower() == 'bilstm':		model = BiLSTM(self.inp_dim, self.num_class, self.feat2dim, self.loss2fact, self.p)
		elif 	self.p.model.lower() == 'multi-bilstm':	model = MultiBiLSTM(self.inp_dim, self.num_class, self.feat2dim, self.loss2fact, self.p)
		elif 	self.p.model.lower() == 'multih-bilstm':model = MultiBiLSTMHier(self.inp_dim, self.num_class, self.feat2dim, self.loss2fact, self.p)
		elif 	self.p.model.lower() == 'bert-bilstm':
			bert_config = BertConfig.from_pretrained(self.p.bert_model)
			model = BertBiLSTM(bert_config, self.inp_dim, self.num_class, self.feat2dim, self.loss2fact, self.p)
		else: raise NotImplementedError

		model = model.to(self.device)

		pprint(model)
		# import pdb; pdb.set_trace()
		return model

	def add_optimizer(self, model, train_dataset_length):
		if self.p.opt == 'adam': 
			if "bert" not in self.p.model:
				return torch.optim.Adam(model.parameters(), lr=self.p.lr, weight_decay=self.p.l2), None
			else:
				warmup_proportion 	= 0.1
				n_train_steps		= int(train_dataset_length / self.p.batch_size ) * self.p.max_epochs
				num_warmup_steps	= int(float(warmup_proportion) * float(n_train_steps))

				param_optimizer		= list(model.named_parameters())
				param_optimizer		= [n for n in param_optimizer if 'pooler' not in n[0]]
				no_decay		= ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

				optimizer_grouped_parameters = [
					{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
					{'params': [p for n, p in param_optimizer if     any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
				]

				optimizer = AdamW(optimizer_grouped_parameters,lr=self.p.lr)
				scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=n_train_steps)
				return optimizer, scheduler
		else: 
			return torch.optim.SGD(model.parameters(),  lr=self.p.lr, weight_decay=self.p.l2), None

	def __init__(self, params):
		self.p = params

		self.save_dir = '{}/{}'.format(self.p.model_dir, self.p.log_db)
		if not os.path.exists(self.p.log_dir): os.system('mkdir -p {}'.format(self.p.log_dir))		# Create log directory if doesn't exist
		if not os.path.exists(self.save_dir):  os.system('mkdir -p {}'.format(self.save_dir))		# Create model directory if doesn't exist

		# Get Logger
		self.logger	= get_logger(self.p.name, self.p.log_dir, self.p.config_dir)
		self.logger.info(vars(self.p)); pprint(vars(self.p))

		if self.p.gpu != '-1' and torch.cuda.is_available():
			self.device = torch.device('cuda')
			torch.cuda.set_rng_state(torch.cuda.get_rng_state())
			torch.backends.cudnn.deterministic = True
		else:
			self.device = torch.device('cpu')

		self.train_dataset_length = self.load_data()
		self.model        = self.add_model()
		self.optimizer,self.scheduler    = self.add_optimizer(self.model, self.train_dataset_length)

	def save_model(self, save_path):
		state = {
			'state_dict'	: self.model.state_dict(),
			'best_test'	: self.best_test,
			'best_val'	: self.best_val,
			'best_epoch'	: self.best_epoch,
			'optimizer'	: self.optimizer.state_dict(),
			'args'		: vars(self.p)
		}
		torch.save(state, '{}/{}'.format(save_path, self.p.name))		

	def load_model(self, load_path):
		state = torch.load('{}/{}'.format(load_path, self.p.name))
		self.best_val		= state['best_val']
		self.best_test		= state['best_test']
		self.best_epoch		= state['best_epoch']

		self.model.load_state_dict(state['state_dict'])
		self.optimizer.load_state_dict(state['optimizer'])

	def get_acc(self, logits, labels):
		all_logits = {k: np.concatenate(mergeList(v), axis=0) for k, v in comb_dict(logits).items()}
		all_labels = {k: np.concatenate(mergeList(v), axis=0) for k, v in comb_dict(labels).items()}

		result = {}
		for key in all_logits.keys():
			logit, label = all_logits[key], all_labels[key]
			if logit.shape == label.shape: 
				# import pdb; pdb.set_trace()
				result['{}_{}'.format(key, 'auc')] = np.round(average_precision_score(label.reshape(-1), logit.reshape(-1), average="micro"), 3)
				try: 	result['{}_{}'.format(key, 'roc')] = np.round(roc_auc_score(label, logit), 3)
				except: result['{}_{}'.format(key, 'roc')] = 0
			else: 	
				result[key] = np.round(f1_score(label, logit.argmax(1), average="macro"), 3)

		return result

	def execute(self, batch):
		batch		= to_gpu(batch, self.device)
		if "bert" not in self.p.model:
			loss, logits 	= self.model(batch['conv_pad'], batch['conv_len'], batch['conv_mask'], feats=batch['feats'], labels=batch['labels'])
		else:   loss, logits 	= self.model(batch['conv_b_token_ids'], batch['conv_len'], batch['conv_b_att_mask'], batch['conv_mask'], labels=batch['labels'])
		return loss, logits

	def predict(self, epoch, split, return_extra=False):
		self.model.eval()

		all_eval_loss, all_logits, all_labels, all_trans, cnt = [], [], [], [], 0

		with torch.no_grad():
			for batches in self.data_iter[split]:
				for k, batch in enumerate(batches):
					eval_loss, logits = self.execute(batch)

					if (k+1) % self.p.log_freq == 0:
						# eval_res = self.get_acc(all_logits, all_labels)

						self.logger.info('[E: {}] | {:.3}% | {} | Eval {} --> Loss: {:.3}'.format(epoch, \
							100*cnt/len(self.data[split]),  self.p.name, split, np.mean(all_eval_loss)))

					all_eval_loss.append(eval_loss.item())

					logits = {k: v.cpu() for k, v in logits.items()}
					logits = {k: [v_i[:batch['conv_len'][i].item()] for i, v_i in enumerate(v)] for k, v in logits.items()}
					labels = {k: [v_i[:batch['conv_len'][i].item()] for i, v_i in enumerate(v)] for k, v in batch['labels'].items() if k != 'chief_comp'}

					all_logits.append(logits)
					all_labels.append(labels)
					all_trans.append(batch['_rest'])

					cnt += batch['conv_len'].shape[0]

		eval_res = self.get_acc(all_logits, all_labels)

		if return_extra:	return np.mean(all_eval_loss), eval_res, all_logits, all_labels, all_trans
		else: 			return np.mean(all_eval_loss), eval_res

	def run_epoch(self, epoch, shuffle=True):
		self.model.train()

		all_train_loss, all_logits, all_labels, cnt = [], [], [], 0

		for batches in self.data_iter['train']:
			for k, batch in enumerate(batches):
				self.optimizer.zero_grad()

				train_loss, logits = self.execute(batch)

				if (k+1) % self.p.log_freq == 0:
					# eval_res = self.get_acc(all_logits, all_labels)

					# self.logger.info('[E: {}] | {:.3}% | {} | L: {:.3}, T: {}, B-V:{}, B-T:{}'.format(epoch, \
					# 	100*cnt/len(self.data['train']), self.p.name, np.mean(all_train_loss), list(eval_res.values()), 
					# 	list(self.best_val.values()), list(self.best_test.values())))

					self.logger.info('[E: {}] | {:.3}% | {} | L: {:.3}, B-V:{}, B-T:{}'.format(epoch, \
						100*cnt/len(self.data['train']), self.p.name, np.mean(all_train_loss), 
						list(self.best_val.values()), list(self.best_test.values())))

				all_train_loss.append(train_loss.item())

				logits = {k: v.cpu() for k, v in logits.items()}
				logits = {k: [v_i[:batch['conv_len'][i].item()] for i, v_i in enumerate(v.detach().cpu().numpy())] for k, v in logits.items()}
				labels = {k: [v_i[:batch['conv_len'][i].item()] for i, v_i in enumerate(v.cpu().numpy())] for k, v in batch['labels'].items() if k != 'chief_comp'}

				all_logits.append(logits)
				all_labels.append(labels)

				# import pdb; pdb.set_trace()

				train_loss.backward()
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.10)
				self.optimizer.step()
				if "bert" in self.p.model:
					self.scheduler.step()

				cnt += batch['conv_len'].shape[0]

		eval_res = self.get_acc(all_logits, all_labels)

		return np.mean(all_train_loss), eval_res

	def fit(self):
		self.best_val, self.best_test, self.best_epoch = {}, {}, 0

		if self.p.restore:
			self.load_model(self.save_dir)
			test_loss, test_acc = self.predict(0, 'test')
			pprint(test_acc)

			import pdb; pdb.set_trace()

			for param in self.model.named_parameters():
				print(param)

			print(sum(p.numel() for p in self.model.parameters() if p.requires_grad))


			if self.p.dump_only:


				all_logits, all_labels, all_trans = [], [], []

				for split in ['test', 'train', 'valid']:
					loss, acc, logits, label, trans = self.predict(0, split, return_extra=True)
					all_logits += logits
					all_labels += label
					all_trans  += trans

				res = {
					'transcript'	: all_trans,
					'labels'	: all_labels,
					'logits'	: all_logits,
					'lbl2id'	: self.lbl2id,
					'tag2id'	: self.tag2id,
				}

				pickle.dump(res, open('./visualize/predictions/{}.pkl'.format(self.p.name), 'wb'))
				exit(0)
			exit(0)

		kill_cnt = 0
		for epoch in range(self.p.max_epochs):
			train_loss, train_acc = self.run_epoch(epoch)
			valid_loss, valid_acc = self.predict(epoch, 'valid')

			if valid_acc[self.p.target] > self.best_val.get(self.p.target, 0.0):
				self.best_val		= valid_acc
				_, self.best_test	= self.predict(epoch, 'test')
				self.best_epoch		= epoch
				self.save_model(self.save_dir)
				kill_cnt = 0
				# import pdb; pdb.set_trace()
				self.logger.info('New learning rate [{}]'.format(self.optimizer.param_groups[0]['lr']))
			else:
				kill_cnt += 1
				self.optimizer.param_groups[0]['lr'] *= self.p.decay_rate
				self.logger.info('New learning rate [{}]'.format(self.optimizer.param_groups[0]['lr']))
				if kill_cnt > 7:
					self.logger.info('Early Stopping!')
					break

			self.logger.info('Epoch [{}] | {} | Summary: Train Loss: {:.3}, Train Acc: {}, Valid Acc: {}, Valid Loss: {:.3}, Best valid: {}, Best Test: {}'
					.format(epoch, self.p.name, train_loss, train_acc, valid_acc, valid_loss, self.best_val, self.best_test))

		self.logger.info('Best Performance: {}'.format(self.best_test)) 

if __name__== "__main__":

	parser = argparse.ArgumentParser(description='MedFilter')

	parser.add_argument('--gpu',      	 default='0',                				help='GPU to use')
	parser.add_argument("--model", 		 default='bilstm', 	type=str, 			help='Model for training and inference')
	parser.add_argument("--embed", 	 	 default="bert_ft", 	type=str, 	help="bert, biobert")

	# Features related
	parser.add_argument("--feat", 	 	 default='speaker,position,semantic', 	type=str, 	help='List of features to be appended in embeddings')
	parser.add_argument('--feat_dim', 	 default='10,20,16', 		   	type=str, 	help='List of dimension of different features wrt --feat')
	parser.add_argument('--feat_cat', 	 default='3,4,133', 			type=str, 	help='List categories in different features wrt --feat')

	parser.add_argument('--chief_model', 	 default='hidden', 			type=str, 	help='Temporary: decides chief model to use')

	parser.add_argument('--target',    	 default='med_class',   				help='Target to predict using the model')
	parser.add_argument('--loss', 		 default='med_tag,med_class', 				help='Loss terms to include')
	parser.add_argument('--loss_fact', 	 default='0.5,0.5', 					help='Loss terms to include')

	# RNN
	parser.add_argument('--rnn_layers',	 default=1, 		type=int, 			help='Number of layers')
	parser.add_argument('--rnn_dim', 	 default=128, 		type=int, 			help='Size of first hidden state')
	parser.add_argument('--rnn_drop', 	 default=0.0, 		type=float, 		help='Dropout')

	# Bert
	parser.add_argument('--max_utt_len', 	 default=64, 		type=int, 			help='Max allowed length of utt')
	parser.add_argument('--max_conv_len', 	 default=128, 		type=int, 			help='Max allowed length if conv in batch')
	parser.add_argument('--bert_model', 	 default='bert-base-uncased', 			type=str, 	help='Which Bert model')

	# ind models
	parser.add_argument('--ind_cat', 	 	default='symptoms', 	type=str, 	help='Which category to check')

	parser.add_argument('--pos_gate', 	 	default=0, 		type=int, 	help='Positional gating')

	parser.add_argument('--decay_rate', 	default=1, 		type=float, 	help='Which category to check')

	parser.add_argument('--epoch',    	 dest='max_epochs',     default=300,    type=int,       help='Max epochs')
	parser.add_argument('--batch',    	 dest='batch_size',     default=16,     type=int,      	help='Batch size')
	parser.add_argument('--batch_factor',    dest='batch_factor',   default=50,     type=int,      	help='Number of batches to generate at one time')
	parser.add_argument('--num_workers',	 type=int,              default=0,                   	help='Number of cores used for preprocessing data')
	parser.add_argument('--opt',      	 default='adam',             				help='Optimizer to use for training')
	parser.add_argument('--lr', 	 	 default=0.001, 	type=float, 			help='The initial learning rate for Adam.')
	parser.add_argument('--l2', 	 	 default=0.0, 		type=float, 			help='The initial learning rate for Adam.')

	parser.add_argument('--log_db',    	 default='test',   	     				help='Experiment name')
	parser.add_argument('--seed',     	 default=1234,   	type=int,       		help='Seed for randomization')
	parser.add_argument('--log_freq',    	 default=10,   		type=int,     			help='Display performance after these number of batches')
	parser.add_argument('--name',     	 default='test',             				help='Name of the run')
	parser.add_argument('--restore',  				action='store_true',        	help='Restore from the previous best saved model')
	parser.add_argument('--dump_only',  				action='store_true',        	help='Dump logits of validation dataset')
	parser.add_argument('--dump_all',  				action='store_true',        	help='Dump logits of validation dataset')
	parser.add_argument('--nosave',  				action='store_true',        	help='Whether to save the best model or not')
	parser.add_argument('--test',  					action='store_true',        	help='Whether to save the best model or not')
	parser.add_argument('--data_split',  				action='store_true',        	help='Whether to save the best model or not')

	parser.add_argument('--data_dir',  	 default='./data/',		help='Directory containing dataset')
	parser.add_argument('--embed_dir',  	 default='./data/',		help='Directory containing embeddings')
	parser.add_argument('--config_dir',   	 default='./project/config',        	help='Config directory')
	parser.add_argument('--model_dir',   	 default='/project/models/',        				help='Model directory')
	parser.add_argument('--log_dir',   	 default='./log',   	   				help='Log directory')
	parser.add_argument('--res_dir',  	 default='/project/resources/seq_labeling',help='Directory containing dataset')

	args = parser.parse_args()
	set_gpu(args.gpu)

	if not args.restore: args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')

	# Set seed
	np.random.seed(args.seed)
	random.seed(args.seed)
	torch.manual_seed(args.seed)

	# Create Model
	model = Main(args)
	model.fit()
	print('Model Trained Successfully!!')