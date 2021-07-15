import sys; sys.path.append('../common')
from helper import *
from base_models import *

# BERT-FT BiLSTM model

class BiLSTM(Base):

	def __init__(self, inp_dim, num_class, feat2dim, loss2fact, config):
		super(BiLSTM, self).__init__(inp_dim, feat2dim, loss2fact, config)

		in_dim = self.inp_dim
		for key in self.loss2fact:
			super(BiLSTM, self).add_module(key + '_classify', LSTMClassifier(in_dim, num_class[key], self.p.rnn_dim, self.p.rnn_layers, self.p.rnn_drop))
			in_dim += self.p.rnn_dim

	def get_attention(self, inp, vec):
		atten_scr = torch.exp((inp * vec).sum(-1, keepdim=True))
		atten_wts = atten_scr / atten_scr.sum(1, keepdim=True)
		return atten_wts

	def forward(self, sent_embed, sent_len, sent_mask, feats=None, labels=None):
		sent_embed = self.add_features(sent_embed, feats)

		model_out, loss	= {}, 0
		for key, fact in self.loss2fact.items():
			model_out[key]	= getattr(self, key + '_classify').forward(sent_embed, sent_len)
			sent_embed 	= torch.cat([sent_embed, model_out[key]['out']], dim=2)
			loss 		+= self.get_loss(model_out[key]['logits'], labels[key], sent_mask) * fact

		logits = {k: v['logits'] for k, v in model_out.items()}
		return loss, logits

# BERT-FT MS-BiLSTM model

class MultiBiLSTM(Base):

	def __init__(self, inp_dim, num_class, feat2dim, loss2fact, config):
		super(MultiBiLSTM, self).__init__(inp_dim, feat2dim, loss2fact, config)

		in_dim = self.inp_dim
		for key in self.loss2fact:
			if key == "med_class":
				super(MultiBiLSTM, self).add_module(key + '_pt_lstm', LSTMOnly(in_dim, num_class[key], self.p.rnn_dim, self.p.rnn_layers, self.p.rnn_drop))
				super(MultiBiLSTM, self).add_module(key + '_dr_lstm', LSTMOnly(in_dim, num_class[key], self.p.rnn_dim, self.p.rnn_layers, self.p.rnn_drop))
				super(MultiBiLSTM, self).add_module(key + '_ot_lstm', LSTMOnly(in_dim, num_class[key], self.p.rnn_dim, self.p.rnn_layers, self.p.rnn_drop))
				super(MultiBiLSTM, self).add_module(key + '_lstm', LSTMOnly(in_dim, num_class[key], self.p.rnn_dim, self.p.rnn_layers, self.p.rnn_drop))
				super(MultiBiLSTM, self).add_module(key + '_linear', nn.Linear(self.p.rnn_dim, num_class[key]))
				
				if self.p.pos_gate:
					super(MultiBiLSTM, self).add_module(key + '_pos_reducer', nn.Linear(self.feat2dim['position'][1], 3))
				self.gate = nn.Parameter(torch.Tensor(3))
				torch.nn.init.uniform_(self.gate)
				# self.comp_gate = nn.Parameter(torch.Tensor(3,self.p.rnn_dim))
			else:
				super(MultiBiLSTM, self).add_module(key + '_lstm', LSTMOnly(in_dim, num_class[key], self.p.rnn_dim, self.p.rnn_layers, self.p.rnn_drop))
				super(MultiBiLSTM, self).add_module(key + '_linear', nn.Linear(self.p.rnn_dim, num_class[key]))
			in_dim += self.p.rnn_dim

	def get_attention(self, inp, vec):
		atten_scr = torch.exp((inp * vec).sum(-1, keepdim=True))
		atten_wts = atten_scr / atten_scr.sum(1, keepdim=True)
		return atten_wts

	def forward(self, sent_embed, sent_len, sent_mask, feats=None, labels=None):
		sent_embed, other_feats = self.add_features(sent_embed, feats, True)

		logits, loss	= {}, 0
		for key, fact in self.loss2fact.items():
			if key == 'med_class':
				out = []

				out.append(getattr(self, key + '_pt_lstm').forward(sent_embed, sent_len)['out']) # batch_size X num_utt X embed_size
				out.append(getattr(self, key + '_dr_lstm').forward(sent_embed, sent_len)['out'])
				out.append(getattr(self, key + '_ot_lstm').forward(sent_embed, sent_len)['out'])

				out_common 		= getattr(self, key + '_lstm').forward(sent_embed, sent_len)['out'] # batch_size X num_utt X embed_size

				# import pdb; pdb.set_trace()

				out 			= torch.stack(out).transpose(0, 1) # batch_size X num_sp X num_utt X embed_size

				batch_arange	= torch.arange(sent_embed.shape[0]).long()

				# import pdb; pdb.set_trace()

				if self.p.pos_gate:
					spk_pos     = getattr(self, key + '_pos_reducer').forward(other_feats['position']).unsqueeze(-1).repeat(1, 1, 1, self.p.rnn_dim) # batch_size X num_utt X num_sp X rnn_dim
					spk_pos     = torch.sigmoid(spk_pos)

				gate			= torch.sigmoid(self.gate).reshape(1, -1, 1).repeat(sent_embed.shape[0], 1, self.p.rnn_dim) # batch_size X num_sp X rnn_dim

				# gate			= torch.sigmoid(self.comp_gate).reshape(1, -1, self.p.rnn_dim).repeat(sent_embed.shape[0], 1, 1) # batch_size X num_sp X rnn_dim

				for u in range(sent_embed.shape[1]):
					utt_gate    = gate[batch_arange, feats['speaker'][:,u]] # batch_size X rnn_dim

					if self.p.pos_gate:
						pos_gate= spk_pos[batch_arange, u, feats['speaker'][:,u], :] # batch_size X rnn_dim
						utt_gate= utt_gate * pos_gate
						

					# import pdb; pdb.set_trace()

					out_u_sp	= out[batch_arange, feats['speaker'][:,u], u, :]
					out_u_com	= out_common[:, u, :]
					out_u 		= utt_gate * out_u_com
					out_u		= out_u + out_u_sp * (1 - utt_gate)
					out_u 		= out_u.unsqueeze(1)
					if u == 0:
						out_final = out_u # batch_size X 1 X embed_size
					else:
						out_final = torch.cat([out_final, out_u], dim = 1) # # batch_size X num_utt X embed_size

				# import pdb; pdb.set_trace()
				# out_final		= out[batch_arange, feats['speakers']] # feats['speakers'] = batch_size X num_utt
				
				logits[key]		= getattr(self, key + '_linear').forward(out_final)
				loss 			+= self.get_loss(logits[key]	, labels[key], sent_mask) * fact
				# import pdb; pdb.set_trace()
			else:
				out				= getattr(self, key + '_lstm').forward(sent_embed, sent_len)['out']

				logits[key]		= getattr(self, key + '_linear').forward(out)

				sent_embed 		= torch.cat([sent_embed, out], dim=2)
				loss 			+= self.get_loss(logits[key], labels[key], sent_mask) * fact
				# import pdb; pdb.set_trace()


		return loss, {k: v for k, v in logits.items()}

# BERT-FT Hier MS-BiLSTM model

class MultiBiLSTMHier(Base):

	def __init__(self, inp_dim, num_class, feat2dim, loss2fact, config):
		super(MultiBiLSTMHier, self).__init__(inp_dim, feat2dim, loss2fact, config)

		in_dim = self.inp_dim
		self.gate1 = nn.Parameter(torch.Tensor(3))
		self.gate2 = nn.Parameter(torch.Tensor(3))
		for key in self.loss2fact:
			super(MultiBiLSTMHier, self).add_module(key + '_pt_lstm', LSTMOnly(in_dim, num_class[key], self.p.rnn_dim, self.p.rnn_layers, self.p.rnn_drop))
			super(MultiBiLSTMHier, self).add_module(key + '_dr_lstm', LSTMOnly(in_dim, num_class[key], self.p.rnn_dim, self.p.rnn_layers, self.p.rnn_drop))
			super(MultiBiLSTMHier, self).add_module(key + '_ot_lstm', LSTMOnly(in_dim, num_class[key], self.p.rnn_dim, self.p.rnn_layers, self.p.rnn_drop))
			super(MultiBiLSTMHier, self).add_module(key + '_lstm', LSTMOnly(in_dim, num_class[key], self.p.rnn_dim, self.p.rnn_layers, self.p.rnn_drop))
			super(MultiBiLSTMHier, self).add_module(key + '_linear', nn.Linear(self.p.rnn_dim, num_class[key]))
			
			if False: #self.p.pos_gate:
				super(MultiBiLSTMHier, self).add_module(key + '_pos_reducer', nn.Linear(self.feat2dim['position'][1], 3))

			# self.gate_1 = nn.Parameter(torch.Tensor(3)))
			# torch.nn.init.uniform_(getattr(self, key + "_gate"))
			# torch.nn.init.uniform_(self.gate[key])
			in_dim += self.p.rnn_dim

	def get_attention(self, inp, vec):
		atten_scr = torch.exp((inp * vec).sum(-1, keepdim=True))
		atten_wts = atten_scr / atten_scr.sum(1, keepdim=True)
		return atten_wts

	def forward(self, sent_embed, sent_len, sent_mask, feats=None, labels=None):
		sent_embed, other_feats = self.add_features(sent_embed, feats, True)

		logits, loss	= {}, 0
		for key, fact in self.loss2fact.items():
			out = []

			out.append(getattr(self, key + '_pt_lstm').forward(sent_embed, sent_len)['out']) # batch_size X num_utt X embed_size
			out.append(getattr(self, key + '_dr_lstm').forward(sent_embed, sent_len)['out'])
			out.append(getattr(self, key + '_ot_lstm').forward(sent_embed, sent_len)['out'])

			out_common 		= getattr(self, key + '_lstm').forward(sent_embed, sent_len)['out'] # batch_size X num_utt X embed_size

			# import pdb; pdb.set_trace()

			out 			= torch.stack(out).transpose(0, 1) # batch_size X num_sp X num_utt X embed_size

			batch_arange	= torch.arange(sent_embed.shape[0]).long()

			# import pdb; pdb.set_trace()

			if False: #self.p.pos_gate:
				spk_pos     = getattr(self, key + '_pos_reducer').forward(other_feats['position']).unsqueeze(-1).repeat(1, 1, 1, self.p.rnn_dim) # batch_size X num_utt X num_sp X rnn_dim
				spk_pos     = torch.sigmoid(spk_pos)

			if key =="med_tag":
				gate			= torch.sigmoid(self.gate1).reshape(1, -1, 1).repeat(sent_embed.shape[0], 1, self.p.rnn_dim) # batch_size X num_sp X rnn_dim
			else:
				gate			= torch.sigmoid(self.gate2).reshape(1, -1, 1).repeat(sent_embed.shape[0], 1, self.p.rnn_dim) # batch_size X num_sp X rnn_dim

			# gate			= torch.sigmoid(self.comp_gate).reshape(1, -1, self.p.rnn_dim).repeat(sent_embed.shape[0], 1, 1) # batch_size X num_sp X rnn_dim

			for u in range(sent_embed.shape[1]):
				utt_gate    = gate[batch_arange, feats['speaker'][:,u]] # batch_size X rnn_dim
				utt_gate	= utt_gate

				if False: #self.p.pos_gate:
					pos_gate= spk_pos[batch_arange, u, feats['speaker'][:,u], :] # batch_size X rnn_dim
					utt_gate= utt_gate * pos_gate
					

				# import pdb; pdb.set_trace()

				out_u_sp	= out[batch_arange, feats['speaker'][:,u], u, :]
				out_u_com	= out_common[:, u, :]
				out_u 		= utt_gate * out_u_com
				out_u		= out_u + out_u_sp * (1 - utt_gate)
				out_u 		= out_u.unsqueeze(1)
				if u == 0:
					out_final = out_u # batch_size X 1 X embed_size
				else:
					out_final = torch.cat([out_final, out_u], dim = 1) # # batch_size X num_utt X embed_size

			# import pdb; pdb.set_trace()
			# out_final		= out[batch_arange, feats['speakers']] # feats['speakers'] = batch_size X num_utt
			
			logits[key]		= getattr(self, key + '_linear').forward(out_final)

			sent_embed 		= torch.cat([sent_embed, out_final], dim=2)

			loss 			+= self.get_loss(logits[key], labels[key], sent_mask) * fact
			# import pdb; pdb.set_trace()

		return loss, {k: v for k, v in logits.items()}