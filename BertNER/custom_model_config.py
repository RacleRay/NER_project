#-*- coding:utf-8 -*-
# author: Racle
# project: BertNER


bert_lstm_crf_config = {
    "hidden_dropout_prob": 0.2,
    "num_layers": 1,
    "lstm_dropout": 0.2,
    "label_encoding": "BIO",
    "freez_prrtrained": False,
    # MDP
    "dp_rate": 0.5,
    "num_dp": 5,

}