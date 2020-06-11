#-*- coding:utf-8 -*-
# author: Racle
# project: BertNER

import torch
from torch import nn
from transformers import (
    PreTrainedModel,
    BertModel,
    BertPreTrainedModel,
    AlbertModel,
    AlbertPreTrainedModel,
    XLNetModel,
    XLNetPreTrainedModel,
    DistilBertConfig,
    DistilBertModel,
    ElectraForMaskedLM,
    ElectraForPreTraining,
    RobertaConfig,
    RobertaModel,
    ElectraConfig,
    ElectraModel,
    ElectraPreTrainedModel,
)
from transformers.modeling_utils import SequenceSummary
from .crf_layer import ConditionalRandomField as crf

class BertLstmCrfMdp(BertPreTrainedModel):
    def __init__(self, config, extra_config, ignore_ids):
        """
        mdp -- 多个dropout
        num_labels : int, required
            Number of tags.
        idx2tag : ``Dict[int, str]``, required
            A mapping {label_id -> label}. Example: {0:"B-LOC", 1:"I-LOC", 2:"O"}
        label_encoding : ``str``, required
            Indicates which constraint to apply. Current choices are
            "BIO", "IOB1", "BIOUL", "BMES" and "BIOES",.
                B = Beginning
                I/M = Inside / Middle
                L/E = Last / End
                O = Outside
                U/W/S = Unit / Whole / Single
        """
        super(BertLstmCrfMdp, self).__init__(config)
        self.pretraind = BertModel(config)
        self.dropout = nn.Dropout(extra_config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.bilstm = nn.LSTM(input_size=config.hidden_size,
                              hidden_size=config.hidden_size // 2,
                              batch_first=True,
                              num_layers=extra_config.num_layers,
                              dropout=extra_config.lstm_dropout,
                              bidirectional=True)
        self.crf = crf(config.num_labels, extra_config.label_encoding, extra_config.idx2tag)
        self.dropouts = nn.ModuleList([
            nn.Dropout(extra_config.dp_rate) for _ in range(extra_config.num_dp)
        ])

        self.init_weights()
        if extra_config.freez_prrtrained:
            for param in self.pretraind.parameters():
                param.requires_grad = False

        self.ignore_ids = ignore_ids

    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                labels=None):
        # outputs的组成：
        # last_hidden_state： Sequence of hidden-states at the output of the last layer of the model.
        #                     (batch_size, sequence_length, hidden_size)
        # pooler_output:      Last layer hidden-state of the first token of the sequence (classification token)
        #                     processed by a Linear layer and a Tanh activation function.
        # hidden_states：     one for the output of the embeddings + one for the output of each layer.
        #                     each is (batch_size, sequence_length, hidden_size)
        # attentions:         Attentions weights after the attention softmax of each layer.
        #                     each is (batch_size, num_heads, sequence_length, sequence_length)
        outputs = self.pretraind(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
        last_hidden_state = outputs[0]

        seq_output = self.dropout(last_hidden_state)
        seq_output, _ = self.bilstm(seq_output)
        seq_output = nn.LayerNorm(seq_output.size()[-1])(seq_output)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                logits = self.classifier(dropout(seq_output))
            else:
                logits += self.classifier(dropout(seq_output))
        logits = logits / len(self.dropouts)

        outputs = (logits,) + outputs[2:]

        masked_labels, masked_logits = self._get_masked_inputs(input_ids, labels, logits, attention_mask)
        if labels is not None:
            loss = self.crf(masked_logits, masked_labels, mask=None)  # mask=None: 已经处理了所有的无用的位置
            outputs = (loss,) + outputs

        # (loss), logits, (hidden_states), (attentions)
        return outputs

    def _get_masked_inputs(self, input_ids, label_ids, logits, attention_mask):
        ignore_ids = self.ignore_ids

        # Remove unuseful positions
        masked_ids = input_ids[(1 == attention_mask)]
        masked_labels = label_ids[(1 == attention_mask)]
        masked_logits = logits[(1 == attention_mask)]
        for id in ignore_ids:
            masked_labels = masked_labels[(id != masked_ids)]
            masked_logits = masked_logits[(id != masked_ids)]
            masked_ids = masked_ids[(id != masked_ids)]

        return masked_labels, masked_logits
