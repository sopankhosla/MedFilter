import sys; sys.path.append('../common')
from helper import *
from base_models import *

import model_clinicalBERT

# BERT BiLSTM FT model 

class BertBiLSTM(BertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)

		self.bert	= BertModel(config)
		self.dropout	= nn.Dropout(config.hidden_dropout_prob)

		self.rnn_dim 	= 512
		self.lstm	= nn.LSTM(config.hidden_size, self.rnn_dim // 2, num_layers=1, bidirectional=True, dropout=0.0, batch_first=True)
		self.classifier	= nn.Linear(self.rnn_dim, config.num_labels)

		self.init_weights()


	def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
		outputs = self.bert(
			input_ids.view(-1, input_ids.shape[-1]),
			attention_mask	= attention_mask.view(-1, input_ids.shape[-1]),
			token_type_ids	= token_type_ids,
			position_ids	= position_ids,
			head_mask	= head_mask,
			inputs_embeds	= inputs_embeds,
		)
		
		bert_out	= outputs[1]
		bert_out	= self.dropout(bert_out)
		rnn_in		= bert_out.view(input_ids.shape[0], input_ids.shape[1], -1)
		lstm_out, final	= self.lstm(rnn_in)
		logits		= self.classifier(lstm_out)
		loss 		= F.binary_cross_entropy_with_logits(logits, labels.float())

		return loss, {'med_class': logits}

# BERT, BERT-FT model

class BertPlainNew(BertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)

		self.bert	= BertModel(config)
		self.dropout	= nn.Dropout(config.hidden_dropout_prob)
		self.classifier	= nn.Linear(config.hidden_size, config.num_labels)

		self.init_weights()


	def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
		outputs = self.bert(
			input_ids.view(-1, input_ids.shape[-1]),
			attention_mask	= attention_mask.view(-1, input_ids.shape[-1]),
			token_type_ids	= token_type_ids,
			position_ids	= position_ids,
			head_mask	= head_mask,
			inputs_embeds	= inputs_embeds,
		)
		
		bert_out	= outputs[1]
		bert_out	= self.dropout(bert_out)
		rnn_in		= bert_out.view(input_ids.shape[0], input_ids.shape[1], -1)
		logits		= self.classifier(rnn_in)
		loss 		= F.binary_cross_entropy_with_logits(logits, labels.float())
		
		return loss, {'med_class': logits}

# Clinical BioBert-FT model

class ClinicalBertPlainNew(BertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)

		self.bert		= BertModel(config)
		self.dropout	= nn.Dropout(config.hidden_dropout_prob)
		self.classifier	= nn.Linear(config.hidden_size, config.num_labels)

		self.init_weights()


	def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
		outputs = self.bert(
			input_ids.view(-1, input_ids.shape[-1]),
			attention_mask	= attention_mask.view(-1, input_ids.shape[-1]),
			token_type_ids	= token_type_ids,
			position_ids	= position_ids,
			head_mask	= head_mask,
			inputs_embeds	= inputs_embeds,
		)
		
		bert_out	= outputs[1]
		bert_out	= self.dropout(bert_out)
		rnn_in		= bert_out.view(input_ids.shape[0], input_ids.shape[1], -1)
		logits		= self.classifier(rnn_in)
		loss 		= F.binary_cross_entropy_with_logits(logits, labels.float())
		
		return loss, {'med_class': logits}
