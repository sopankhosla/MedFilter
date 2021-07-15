import sys; sys.path.append('../common'); 
from helper import *
import math

class Base(nn.Module):

	def __init__(self, inp_dim, feat2dim, loss2fact, p):
		super().__init__()
		self.p		= p
		self.feat2dim 	= feat2dim
		self.loss2fact 	= loss2fact
		self.inp_dim 	= inp_dim

		for key, (num_dim, num_cat) in self.feat2dim.items():
			if "logs" not in key:
				if num_dim == num_cat:
					super(Base, self).add_module(key + '_embedding', nn.Embedding(num_cat, num_dim,  padding_idx=None))
					getattr(self, key + '_embedding').weight = nn.Parameter(torch.eye(num_dim), requires_grad=False)
				else:
					super(Base, self).add_module(key + '_embedding', nn.Embedding(num_cat, num_dim,  padding_idx=None))


			self.inp_dim += num_dim

	def get_loss(self, logits, labels, mask=None):
		if logits.shape == labels.shape:
			float_mask = mask.unsqueeze(-1).float()
			loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
			return (loss * mask.unsqueeze(-1)).sum() / mask.sum()
		elif mask is None:
			return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))

		else:
			loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction='none')
			return (loss * mask.reshape(-1)).sum() / mask.sum()

	def add_features(self, sent_embed, feats, return_feats = False):
		app_feats = {}

		for key, ind in feats.items():
			if key not in self.feat2dim: continue
			if "logs" in key:
				sent_embed	= torch.cat([sent_embed, ind], dim=2)
			elif key == 'semantic':
				feat_rep	= getattr(self, key + '_embedding')(ind[0])
				mask 		= ind[1].float()
				feat_masked	= feat_rep * mask.unsqueeze(3)
				# TODO: Attention based weighted sum
				feat_final 	= feat_masked.sum(2) / (mask.sum(2, keepdim=True) + 0.000001)
				sent_embed	= torch.cat([sent_embed, feat_final], dim=2)
			else:
				feat_rep	= getattr(self, key + '_embedding')(ind)
				sent_embed	= torch.cat([sent_embed, feat_rep], dim=2)

			app_feats[key]	= feat_rep

		if return_feats:
			return sent_embed, app_feats
		return sent_embed

class FuseEmbedding(torch.nn.Module):
	def __init__(self, att_dim):
		super(FuseEmbedding, self).__init__()

		self.attention  = ScaledDotProductAttention(att_dim)


	def forward(self, embed, gru_out, hier_type, sent_mask = None):
		l_cont, r_cont			= gru_out.chunk(2, -1)
		fused_embed				= gru_out

		# import pdb; pdb.set_trace()

		if hier_type	== "higru-f":
			fused_embed	= torch.cat([l_cont, embed, r_cont], dim = -1)
		elif hier_type	== "higru-sf":
			l_cont_att	= self.attention(l_cont, l_cont, l_cont, attn_mask = sent_mask)
			r_cont_att	= self.attention(r_cont, r_cont, r_cont, attn_mask = sent_mask)
			fused_embed	= torch.cat([l_cont_att[0], l_cont, embed, r_cont, r_cont_att[0]], dim = -1)

		return fused_embed

class LSTMOnly(nn.Module):
	def __init__(self, inp_dim, out_dim, rnn_dim, rnn_layers, rnn_drop):
		super(LSTMOnly, self).__init__()
		self.lstm	= nn.LSTM(inp_dim, rnn_dim // 2, num_layers=rnn_layers, bidirectional=True, dropout=rnn_drop)

	def forward(self, inp_feat, inp_len):
		packed		= pack_padded_sequence(inp_feat, inp_len, batch_first=True)
		lstm_out, final	= self.lstm(packed)
		lstm_out, _	= pad_packed_sequence(lstm_out, batch_first=True)

		return {
			'out'	: lstm_out,
			'final'	: final
		}


class LSTMClassifier(nn.Module):
	def __init__(self, inp_dim, out_dim, rnn_dim, rnn_layers, rnn_drop):
		super(LSTMClassifier, self).__init__()
		self.lstm	= nn.LSTM(inp_dim, rnn_dim // 2, num_layers=rnn_layers, bidirectional=True, dropout=rnn_drop)
		self.classifier	= nn.Linear(rnn_dim, out_dim)

	def forward(self, inp_feat, inp_len):
		packed		= pack_padded_sequence(inp_feat, inp_len, batch_first=True)
		lstm_out, final	= self.lstm(packed)
		lstm_out, _	= pad_packed_sequence(lstm_out, batch_first=True)

		logits  = self.classifier(lstm_out)

		return {
			'logits': logits,
			'out'	: lstm_out,
			'final'	: final
		}
