import sys; sys.path.append('/projects/symptom_detection/seq_pred');
from bert_models import *

import sys; sys.path.append('../common'); 
from helper import *
from dataloader import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score

import model_clinicalBERT

class Main(object):

	def load_data(self):

		self.data   = {'train': [], 'valid': [], 'test': []}
		
		self.tag2id = {
			'symptoms'			: 0,
			'chief_complaint'	: 1,
			'medications'		: 2,
			'prescription'		: 2,
		}

		self.num_class = {
			'med_tag'	: 2,
			'med_class'	: len(set(self.tag2id.values()))
		}

		self.speaker2id		= {'PT': 0, 'DR': 1, 'REST': 2}
		self.umls_map		= pickle.load(open('{}/features/umls_embed.pkl'.format(self.p.data_dir), 'rb'))
		self.semantic_map	= pickle.load(open('{}/features/semantic.pkl'.format(self.p.data_dir), 'rb'))
		
		self.feat2dim		= OrderedDict(zip(self.p.feat.split(','), zip([int(x) for x in self.p.feat_dim.split(',')], [int(x) for x in self.p.feat_cat.split(',')])))
		self.loss2fact		= OrderedDict(zip(self.p.loss.split(','), [float(x) for x in self.p.loss_fact.split(',')]))

		cache_file = '{}/bert_main-{}.pkl'.format(self.p.cache_dir, self.p.embed)
		if not self.p.cache or not os.path.exists(cache_file):
			for line in tqdm(open('{}/{}'.format(self.p.data_dir, self.p.data_file))):
				conv			= json.loads(line)
				_id			= conv['meta']['id']
				_, conv['transcript']	= zip(*sorted(conv['transcript'].items(), key = lambda x: int(x[0])))
				num_utter 		= len(conv['transcript'])

				# Get Features to be used
				if 'speaker'  in self.feat2dim: conv['speaker']  = [self.speaker2id.get(x['speaker'], self.speaker2id['REST']) for x in conv['transcript']]
				if 'position' in self.feat2dim: conv['position'] = np.int32(mergeList([i+np.zeros(len(x)) for i, x in enumerate(partition(range(num_utter), self.feat2dim['position'][1]))]))
				if 'semantic' in self.feat2dim: conv['semantic'] = self.semantic_map[_id]

				# Get Medical Class
				med_tag		= np.zeros(num_utter)
				med_class	= np.zeros((num_utter, self.num_class['med_class']))
				for mtype, mentions in conv['spans'].items():
					if mtype not in self.tag2id: continue
					for ele in mentions:
						med_tag[np.int32(ele['span'])] = 1
						med_class[np.int32(ele['span']), self.tag2id[mtype]] = 1

				self.data[conv['split']].append(conv)

			self.parallel_tokenize()
			pickle.dump(self.data, open(cache_file, 'wb'))

		else:
			self.data = pickle.load(open(cache_file, 'rb'))

		# Already included in conv['embed'], don't need to handle it separately
		if 'umls' in self.feat2dim: del self.feat2dim['umls']
		self.split_long_conv()

		self.logger.info('\nDataset size -- Train: {}, Valid: {}, Test:{}'.format(len(self.data['train']), len(self.data['valid']), len(self.data['test'])))

		self.logger.info('\nnum_classes: {}'.format(self.num_class))


		def get_data_loader(split, shuffle=True):
			dataset = BertFineDataset(self.data[split], self.num_class, self.p)
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


	def parallel_tokenize(self):
		self.logger.info('Started Parallel tokenization')
		all_data = self.data['train'] + self.data['test'] + self.data['valid']

		def proc_text(conv_list, tokenizer, no_spk):
			out_list  = []
			for conv in conv_list:
				if not no_spk:
					conv['text'] = [tokenizer.convert_tokens_to_ids(['[CLS]'] + tokenizer.tokenize(trans['speaker'] + ": " + trans['txt']) + ['[SEP]']) for trans in conv['transcript']]
				else:
					conv['text'] = [tokenizer.convert_tokens_to_ids(['[CLS]'] + tokenizer.tokenize(trans['txt']) + ['[SEP]']) for trans in conv['transcript']]
				out_list.append(conv)
			return out_list

		if self.p.model.lower() == 'clibert':
			model_dir = "./clinicalBERT/biobert_pretrain_output_disch_100000/"
			tokenizer = BertTokenizer.from_pretrained(model_dir + "vocab.txt")
		else:
			tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

		num_procs	= 10
		chunks		= partition(all_data, num_procs)
		res_list	= mergeList(Parallel(n_jobs = num_procs)(delayed(proc_text)(chunk, tokenizer, self.p.no_spk) for chunk in chunks))
		split_list	= [len(self.data['train']), len(self.data['train']) + len(self.data['test'])]

		self.data['train'], self.data['test'], self.data['valid'] = [res_list[i:j] for i, j in zip([0] + split_list, split_list + [None])]
		self.logger.info('Parallel tokenization over')

	def split_long_conv(self):
		print('Original Count: Train: {}, Valid: {}, Test:{}'.format(len(self.data['train']), len(self.data['valid']), len(self.data['valid'])))
		for split in ['train', 'valid', 'test']:
			for i in range(len(self.data[split])-1, -1, -1):
				conv	  = self.data[split][i]
				num_utter = len(conv['text'])

				if num_utter > self.p.max_utter:
					num_part = int(np.ceil(num_utter / self.p.max_utter))
					for k in range(num_part):
						start_ind	= k * self.p.max_utter
						end_ind		= min( (k+1) * self.p.max_utter, num_utter)

						sub_conv = {}
						sub_conv['transcript']	= conv['transcript'][start_ind: end_ind]
						sub_conv['text']	= conv['text'][start_ind: end_ind]
						sub_conv['meta']	= conv['meta']
						sub_conv['split']	= conv['split']
						sub_conv['labels'] 	= {k: v[start_ind: end_ind] for k, v in conv['labels'].items()}
						
						self.data[split].append(sub_conv)

					del self.data[split][i]

			for i in range(len(self.data[split])-1, -1, -1):
				self.data[split][i]['text'] = [[x[0]] + x[1:min(self.p.max_seq_len-1, len(x)-1)] + [x[-1]] for x in self.data[split][i]['text']]
				# import pdb; pdb.set_trace()
				assert self.data[split][i]['text'][0][-1] == 102
				assert self.data[split][i]['text'][0][0] == 101
				assert len(self.data[split][i]['text'][0]) <= self.p.max_seq_len

		print('Updated Count: Train: {}, Valid: {}, Test:{}'.format(len(self.data['train']), len(self.data['valid']), len(self.data['valid'])))

	def add_model(self):
		if   self.p.model.lower() == 'bert':  	    model = BertPlainNew.from_pretrained('bert-base-uncased', num_labels=self.num_class[self.p.target], output_attentions=False, output_hidden_states=False)
		elif   self.p.model.lower() == 'clibert':
			model_dir = "./clinicalBERT/biobert_pretrain_output_disch_100000/"
			model = ClinicalBertPlainNew.from_pretrained(model_dir, num_labels=self.num_class[self.p.target], output_attentions=False, output_hidden_states=False)
			# model = ClinicalBertPlainNew(bert_config)
		elif self.p.model.lower() == 'bert-bilstm': model = BertBiLSTM.from_pretrained('bert-base-uncased', num_labels=self.num_class[self.p.target], output_attentions=False, output_hidden_states=False)
		else: raise NotImplementedError

		model = model.to(self.device)
		return model

	def add_optimizer(self, parameters):
		if   self.p.opt == 'adam': 
			param_optimizer	= list(self.model.named_parameters())
			param_optimizer	= [n for n in param_optimizer if 'pooler' not in n[0]]
			no_decay	= ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
			optimizer_grouped_parameters = [
				{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
				{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
			]

			return AdamW(optimizer_grouped_parameters, lr=self.p.lr)

		elif self.p.opt == 'adam_old' 	: return torch.optim.Adam(parameters, lr=self.p.lr, weight_decay=self.p.l2)
		else                    	: return torch.optim.SGD(parameters,  lr=self.p.lr, weight_decay=self.p.l2)

	def __init__(self, params):
		self.p = params

		self.save_dir = '{}/{}/{}'.format(self.p.model_dir, self.p.log_db, self.p.name)
		if not os.path.exists(self.p.log_dir): os.system('mkdir -p {}'.format(self.p.log_dir))		# Create log directory if doesn't exist
		if not os.path.exists(self.save_dir):  os.system('mkdir -p {}'.format(self.save_dir))		# Create model directory if doesn't exist

		# Get Logger
		self.mongo_log	= ResultsMongo(self.p)
		self.logger	= get_logger(self.p.name, self.p.log_dir, self.p.config_dir)
		self.logger.info(vars(self.p)); pprint(vars(self.p))

		if self.p.gpu != '-1' and torch.cuda.is_available():
			self.device = torch.device('cuda')
			torch.cuda.set_rng_state(torch.cuda.get_rng_state())
			torch.backends.cudnn.deterministic = True
		else:
			self.device = torch.device('cpu')

		self.load_data()
		self.model        = self.add_model()
		self.optimizer    = self.add_optimizer(self.model.parameters())

		num_train_opt_steps	= int(len(self.data['train']) / self.p.batch_size ) * self.p.max_epochs
		num_warmup_steps	= int(float(self.p.warmup_frac) * float(num_train_opt_steps))
		self.scheduler		= get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_opt_steps)

	def save_model(self, save_path):
		state = {
			'state_dict'	: self.model.state_dict(),
			'best_test'	: self.best_test,
			'best_val'	: self.best_val,
			'best_epoch'	: self.best_epoch,
			'optimizer'	: self.optimizer.state_dict(),
			'args'		: vars(self.p)
		}

		torch.save(state, '{}/model.bin'.format(save_path))
		torch.save(self.model.state_dict(), '{}/pytorch.bin'.format(save_path))
		self.model.config.to_json_file('{}/config.json'.format(save_path))

	def load_model(self, load_path):
		state = torch.load('{}/model.bin'.format(load_path))
		self.best_val	= state['best_val']
		self.best_test	= state['best_test']
		self.best_epoch	= state['best_epoch']

		self.model.load_state_dict(state['state_dict'])
		self.optimizer.load_state_dict(state['optimizer'])

	def get_acc(self, logits, labels, mask=None):

		assert self.p.batch_size == 1

		logits = [{k: v[0] for k, v in x.items()} for x in logits]
		labels = [{k: v[0] for k, v in x.items()} for x in labels]

		all_logits = {k: np.concatenate(v, axis=0) for k, v in comb_dict(logits).items()}
		all_labels = {k: np.concatenate(v, axis=0) for k, v in comb_dict(labels).items()}
		

		result = {}
		for key in all_logits.keys():
			logit, label = all_logits[key], all_labels[key]
			result[key] = np.round(average_precision_score(label.reshape(-1), logit.reshape(-1)), 3)

		return result


	def execute(self, batch):
		batch	     = to_gpu(batch, self.device)
		loss, logits = self.model(
					input_ids 	= batch['tok_pad'], 
					attention_mask	= batch['tok_mask'],
					labels		= batch['labels'][self.p.target],
				)
		# acc = {'med_class': self.get_acc(logits, batch['labels'][self.p.target])}
		# return loss, acc, logits
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
					logits = {self.p.target: logits[self.p.target].detach().cpu().numpy()}
					labels = {self.p.target: batch['labels'][self.p.target]}

					all_logits.append(logits)
					all_labels.append(labels)
					# all_trans.append(batch['_rest'])

					cnt += len(batch)

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

					self.logger.info('[E: {}] | {:.3}% | {} | L: {:.3}, B-V:{}, B-T:{}'.format(epoch, \
						100*cnt/len(self.data['train']), self.p.name, np.mean(all_train_loss), 
						list(self.best_val.values()), list(self.best_test.values())))

				all_train_loss.append(train_loss.item())

				logits = {k: v.cpu() for k, v in logits.items()}
				logits = {self.p.target: logits[self.p.target].detach().cpu().numpy()}
				labels = {self.p.target: batch['labels'][self.p.target]}

				all_logits.append(logits)
				all_labels.append(labels)

				train_loss.backward()
				self.optimizer.step()
				if "bert" in self.p.model:
					self.scheduler.step()

				cnt += len(batch)


		eval_res = self.get_acc(all_logits, all_labels)

		return np.mean(all_train_loss), eval_res

	def fit(self):
		self.best_val, self.best_test, self.best_epoch = {}, {}, 0

		if self.p.restore:
			self.load_model(self.save_dir)
			test_loss, test_acc = self.predict(0, 'test')
			pprint(test_acc)

			if self.p.dump_only:

				all_logits, all_labels, all_trans = [], [], []

				for split in ['test', 'train', 'valid']:
					loss, acc, logits, label, trans = self.predict(0, split, return_extra=True)
					all_logits += mergeList(logits)
					all_labels += mergeList(label)
					all_trans  += mergeList(trans)

				res = {
					'transcript'	: all_trans,
					'labels'	: all_labels,
					'logits'	: all_logits,
					'lbl2id'	: self.lbl2id,
					'tag2id'	: self.tag2id,
				}

				# res = {
				# 	'data'		: self.data['test'],
				# 	'transcript'	: mergeList(valid_trans),
				# 	'labels'	: mergeList(valid_y),
				# 	'logits'	: mergeList(valid_logits),
				# 	'acc'		: valid_acc,
				# 	'lbl2id'	: self.lbl2id,
				# 	'tag2id'	: self.tag2id,
				# }
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
			else:
				kill_cnt += 1
				if kill_cnt > 30:
					self.logger.info('Early Stopping!')
					break

			self.logger.info('Epoch [{}] | {} | Summary: Train Loss: {:.3}, Train Acc: {}, Valid Acc: {}, Valid Loss: {:.3}, Best valid: {}, Best Test: {}'
					.format(epoch, self.p.name, train_loss, train_acc, valid_acc, valid_loss, self.best_val, self.best_test))
			self.mongo_log.add_results(self.best_val, self.best_test, self.best_epoch, train_loss)

		self.logger.info('Best Performance: {}'.format(self.best_test))  

if __name__== "__main__":

	parser = argparse.ArgumentParser(description='MedFilter')

	parser.add_argument('--gpu',      	 default='0',                				help='GPU to use')
	parser.add_argument("--model", 		 default='bert', 	type=str, 			help='Model for training and inference')
	parser.add_argument("--embed", 	 	 default="bert", 	type=str, 			help="bert, biobert")

	parser.add_argument('--max_seq_len', 	 default=64, 		type=int, 			help='Number of layers')
	parser.add_argument("--max_utter",   	default=64, 		type=int, 			help="The maximum total input sequence length after WordPiece tokenization") 

	parser.add_argument('--mask_unk',  				action='store_true',        	help='Mask [UNK]')

	parser.add_argument('--no_spk',  				action='store_true',        	help='Don\'t add speaker info in BERT')

	# Features related
	parser.add_argument("--feat", 	 	 default='none', 	type=str, 			help='List of features to be appended in embeddings')
	parser.add_argument('--feat_dim', 	 default='0', 		type=str, 			help='List of dimension of different features wrt --feat')
	parser.add_argument('--feat_cat', 	 default='0', 		type=str, 			help='List categories in different features wrt --feat')

	parser.add_argument('--chief_model', 	 default='hidden', 	type=str, 			help='Temporary: decides chief model to use')

	parser.add_argument('--target',    	 default='med_class',   				help='Target to predict using the model')
	parser.add_argument('--loss', 		 default='med_tag,med_class', 				help='Loss terms to include')
	parser.add_argument('--loss_fact', 	 default='0.5,0.5', 					help='Loss terms to include')

	parser.add_argument('--rnn_layers', 	 default=1, 		type=int, 			help='Number of layers')
	parser.add_argument('--rnn_dim', 	 default=1024, 		type=int, 			help='Size of first hidden state')
	parser.add_argument('--rnn_drop', 	 default=0.0, 		type=float, 			help='Dropout')

	parser.add_argument('--epoch',    	 dest='max_epochs',     default=300,    type=int,       help='Max epochs')
	parser.add_argument('--batch',    	 dest='batch_size',     default=1,     	type=int,      	help='Batch size')
	parser.add_argument('--batch_factor',    dest='batch_factor',   default=50,     type=int,      	help='Number of batches to generate at one time')
	parser.add_argument('--num_workers',	 type=int,              default=0,                   	help='Number of cores used for preprocessing data')
	parser.add_argument('--opt',      	 default='adam',             				help='Optimizer to use for training')
	parser.add_argument('--lr', 	 	 default=1e-5, 	type=float, 			help='The initial learning rate for Adam.')
	parser.add_argument('--l2', 	 	 default=0.0, 		type=float, 			help='The initial learning rate for Adam.')
	parser.add_argument('--warmup_frac', 	 default=0.1, 		type=float, 			help='The initial learning rate for Adam.')

	parser.add_argument('--log_db',    	 default='test',   	     				help='Experiment name')
	parser.add_argument('--seed',     	 default=1234,   	type=int,       		help='Seed for randomization')
	parser.add_argument('--log_freq',    	 default=10,   		type=int,     			help='Display performance after these number of batches')
	parser.add_argument('--name',     	 default='test',             				help='Name of the run')
	parser.add_argument('--restore',  				action='store_true',        	help='Restore from the previous best saved model')
	parser.add_argument('--dump_only',  				action='store_true',        	help='Dump logits of validation dataset')
	parser.add_argument('--dump_all',  				action='store_true',        	help='Dump logits of validation dataset')
	parser.add_argument('--nosave',  				action='store_true',        	help='Whether to save the best model or not')
	parser.add_argument('--cache',  				action='store_true',        	help='Whether to save the best model or not')

	parser.add_argument('--data_dir',  	 default='/data/',							help='Directory containing dataset')
	parser.add_argument('--data_file',  	 default='main.json',						help='File containing dataset')
	parser.add_argument('--config_dir',   	 default='./config',        	help='Config directory')
	parser.add_argument('--model_dir',   	 default='./models',        				help='Model directory')
	parser.add_argument('--log_dir',   	 default='./log',   	   				help='Log directory')
	parser.add_argument('--res_dir',  	 default='./resources/seq_labeling',help='Directory containing dataset')
	parser.add_argument("--cache_dir",  	 default="./cache",					help='Directory containing dataset')

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