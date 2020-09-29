#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   config.py
@Author  :   Racle
@Version :   1.0
@Desc    :   None
'''

import os
import tensorflow as tf


flags = tf.app.flags
root_path = os.getcwd() + os.sep

# configurations when train from scratch
flags.DEFINE_boolean("clean",       True,      "clean train folder")

# configurations for the model
flags.DEFINE_string("model_type",   "idcnn",    "Model type, can be idcnn or bilstm")
flags.DEFINE_integer("seg_dim",     20,         "Embedding size for segmentation, 0 if not used")
flags.DEFINE_integer("char_dim",    100,        "Embedding size for characters")
flags.DEFINE_integer("lstm_dim",    100,        "Num of hidden units in LSTM, or num of filters in IDCNN")
flags.DEFINE_list("dilate_rate",    [1, 1, 2],  "IDCNN dilate rate of 3 layers")
flags.DEFINE_integer("filter_width",3,          "CNN filter width")
flags.DEFINE_integer("num_filter",  100,        "CNN output channels")
flags.DEFINE_integer("num_sublayers", 4,        "Num of IDCNN modules")

# configurations for training
flags.DEFINE_boolean("train",       True,       "Whether train the model")
flags.DEFINE_integer("max_epoch",   100,        "maximum training epochs")
flags.DEFINE_integer("steps_check", 100,        "steps per checkpoint")
flags.DEFINE_float("clip",          5,          "Gradient clip")
flags.DEFINE_float("dropout",       0.5,        "Dropout rate")
flags.DEFINE_float("batch_size",    64,         "batch size")
flags.DEFINE_float("evl_batch_size",100,        "test batch size")
flags.DEFINE_float("lr",            0.001,      "Initial learning rate")
flags.DEFINE_string("optimizer",    "adam",     "Optimizer for training")
flags.DEFINE_boolean("pre_emb",     True,       "Wither use pre-trained embedding")
flags.DEFINE_boolean("zeros",       True,       "Wither replace digits with zero")
flags.DEFINE_boolean("lower",       False,      "Wither lower case")

# configurations for input data
flags.DEFINE_string("emb_file",     os.path.join(root_path+"data", "vec.txt"),        "Path for pre_trained embedding")
flags.DEFINE_string("train_file",   os.path.join(root_path+"data", "example.train"),  "Path for train data")
flags.DEFINE_string("dev_file",     os.path.join(root_path+"data", "example.dev"),    "Path for dev data")
flags.DEFINE_string("test_file",    os.path.join(root_path+"data", "example.test"),   "Path for test data")
# configurations for tagging format
flags.DEFINE_string("tag_schema",   "iobes",    "tagging schema iobes or iob")

# configurations for saving data
flags.DEFINE_string("ckpt_path",    "ckpt",         "Path to save model")
flags.DEFINE_string("summary_path", "summary",      "Path to store summaries")
flags.DEFINE_string("log_file",     "train.log",    "File for log")

flags.DEFINE_string("vocab_file",   "vocab.json",   "File for vocab")
flags.DEFINE_string("map_file",     os.path.join(root_path+"data", "maps.pkl"),        "file for maps")
flags.DEFINE_string("tag_id_map",   os.path.join(root_path+"data", "tag_to_id.txt"),   "Record tag id map")
flags.DEFINE_string("id_tag_map",   os.path.join(root_path+"data", "id_to_tag.txt"),   "Record id tag map")
flags.DEFINE_string("config_file",  os.path.join(root_path+"config", "config_file"),   "File for config")

# configurations for evaluation
flags.DEFINE_string("result_path",  "result",       "Path for results")


FLAGS = tf.app.flags.FLAGS
# check
assert FLAGS.clip < 5.1, "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.lr > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]