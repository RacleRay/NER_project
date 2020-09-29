#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Author  :   Racle
@Version :   1.0
@Desc    :   None
'''

import os
import pickle
import itertools
import json
import logging
import tensorflow as tf
import numpy as np

import utils
import data_manager as dm

from collections import OrderedDict
from model import Model
from config import FLAGS


### Logger settings ###
log_path = os.path.join("log", FLAGS.log_file)

logger = logging.getLogger(FLAGS.log_file)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

fh = logging.FileHandler(FLAGS.log_file)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)


# limit GPU memory
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True


class Runner:
    def __init__(self):
        self.load_data()
        self.load_maps()
        print("\n => Data loading finished. \n")
        self.processs_model_input()
        print("\n => Data preparing finished. \n")

        self.check_pathes()
        self.show_config()

    def load_data(self):
        self.train_sentences = dm.load_sentences(FLAGS.train_file, FLAGS.zeros)
        self.dev_sentences = dm.load_sentences(FLAGS.dev_file, FLAGS.zeros)
        self.test_sentences = dm.load_sentences(FLAGS.test_file, FLAGS.zeros)

        # 转换为IOBES
        dm.update_tag_scheme(self.train_sentences, FLAGS.tag_schema)
        dm.update_tag_scheme(self.dev_sentences, FLAGS.tag_schema)
        dm.update_tag_scheme(self.test_sentences, FLAGS.tag_schema)

    def load_maps(self):
        "函数运行得到目标数据： self.ch_2_id, self.id_2_ch, self.tag_2_id, self.id_2_tag"
        # id 与 字 的映射
        if not os.path.isfile(FLAGS.map_file):
            # create dictionary for word
            if FLAGS.pre_emb:  # 使用预训练词向量中的字进行扩充
                dictionary = dm.char_mapping(self.train_sentences,
                                             FLAGS.lower)[0]
                aug_dict, self.ch_2_id, self.id_2_ch = dm.augment_with_pretrained(
                    dictionary,
                    FLAGS.emb_file,
                    itertools.chain.from_iterable(
                        [[w[0] for w in s] for s in self.test_sentences])
                )
            else:
                _, self.ch_2_id, self.id_2_ch = dm.char_mapping(self.train_sentences,
                                                                FLAGS.lower)
            # Create a dictionary and a mapping for tags
            _, self.tag_2_id, self.id_2_tag = dm.tag_mapping(self.train_sentences,
                                                             FLAGS.tagId_path,
                                                             FLAGS.idTag_path)
            # save
            with open(FLAGS.map_file, 'wb') as f:
                pickle.dump([self.ch_2_id, self.id_2_ch, self.tag_2_id, self.id_2_tag], f)
        else:
            # load
            with open(FLAGS.map_file, 'rb') as f:
                self.ch_2_id, self.id_2_ch, self.tag_2_id, self.id_2_tag = pickle.load(f)

        self.num_chars = len(self.ch_2_id)
        self.num_tags = len(self.tag_2_id)

    def processs_model_input(self):
        "转换文字和tags为 index 输入模型"
        # batch data
        self.train_batcher = dm.BatchManager(self.train_sentences, FLAGS.batch_size,
            self.ch_2_id, self.tag_2_id, FLAGS.lower, FLAGS.train)
        self.dev_bathcer = dm.BatchManager(self.dev_sentences, FLAGS.evl_batch_size,
            self.ch_2_id, self.tag_2_id, FLAGS.lower, FLAGS.train)
        self.test_batcher = dm.BatchManager(self.test_sentences, FLAGS.evl_batch_size,
            self.ch_2_id, self.tag_2_id, FLAGS.lower, FLAGS.train)
        print("%i / %i / %i sentences in train / dev / test." % (
            self.train_batcher.num_data,
            self.dev_bathcer.num_data,
            self.test_batcher.num_data))

    @staticmethod
    def check_pathes():
        if not os.path.isdir(FLAGS.result_path):
            os.makedirs(FLAGS.result_path)
        if not os.path.isdir(FLAGS.ckpt_path):
            os.makedirs(FLAGS.ckpt_path)
        if not os.path.isdir("log"):
            os.makedirs("log")
        logger.info("=> Pathes check finished.")

    def show_config(self):
        if os.path.isfile(FLAGS.config_file):
            with open(FLAGS.config_file, "r", encoding="utf8") as f:
                config = json.load(f)
        else:
            config = OrderedDict()
            config["model_type"] = FLAGS.model_type
            config["num_chars"] = self.num_chars
            config["char_dim"] = FLAGS.char_dim
            config["num_tags"] = self.num_tags
            config["seg_dim"] = FLAGS.seg_dim
            config["lstm_dim"] = FLAGS.lstm_dim
            config["dilate_rate"] = FLAGS.dilate_rate
            config["filter_width"] = FLAGS.filter_width
            config["num_filter"] = FLAGS.num_filter
            config["num_sublayers"] = FLAGS.num_sublayers
            config["batch_size"] = FLAGS.batch_size
            config["emb_file"] = FLAGS.emb_file
            config["clip"] = FLAGS.clip
            config["dropout"] = FLAGS.dropout
            config["optimizer"] = FLAGS.optimizer
            config["lr"] = FLAGS.lr
            config["tag_schema"] = FLAGS.tag_schema
            config["pre_emb"] = FLAGS.pre_emb
            config["zeros"] = FLAGS.zeros
            config["lower"] = FLAGS.lower
            with open(FLAGS.config_file, "w", encoding="utf8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
        self.config = config
        for k, v in config.items():
            logger.info("{}:\t{}".format(k.ljust(15), v))

    def train(self):
        steps_per_epoch = self.train_batcher.num_batches

        with tf.Session(config=tf_config) as sess:
            ### 1. create model and load parameters
            self.model = Model(self.config)
            ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_path)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                self.model.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                logger.info("Created model with fresh parameters.")
                sess.run(tf.global_variables_initializer())
                if self.config["pre_emb"]:  # load pre-trained word vec params
                    # load word vec
                    embed_weights = sess.run(self.model.char_lookup.read_value())
                    embed_weights = utils.load_word2vec(FLAGS.emb_path,
                                                        self.id_2_ch,
                                                        FLAGS.char_dim,
                                                        embed_weights)
                    sess.run(self.model.char_lookup.assign(embed_weights))
                    logger.info("Loaded pre-trained embedding.")

            ### 2. training
            logger.info(" => Start training...")
            loss = []
            with tf.device("/gpu:0"):
                for i in range(FLAGS.epochs):
                    for batch in self.train_batcher.iter_batch(shuffle=False):
                        step, batch_loss = self.model.run_step()
                        loss.append(batch_loss)
                        if step % FLAGS.steps_check == 0:
                            iteration = step // steps_per_epoch + 1
                            logger.info(">>> Epoch:{}, iteration:{}, step:{}/{}, Batch mean loss:{:>9.6f}".format(
                                i+1, iteration, step%steps_per_epoch, steps_per_epoch, np.mean(loss)))
                            loss = []
                    # evaluate at dev set every epoch
                    self.eval(sess, "dev", self.dev_bathcer)
                    # save model
                    if i % 8 == 0:
                        self.model.save_model(sess, FLAGS.ckpt_path, name="train_ner.ckpt")
                        logger.info("=> Model saved. ")
                # evaluate at test set
                self.eval(sess, "test", self.test_batcher)

    def eval(self, sess, mode, data_manager):
        logger.info("=> Evaluate mode: {}".format(mode))
        # evaluate result
        ner_res = self.model.evaluate(sess, data_manager, self.id_2_tag)
        report = utils.test_ner(ner_res, FLAGS.result_path)
        for line in report:
            logger.info(line)
        # score: find best score model on dev set!!!
        f1 = float(report[1].strip().split()[-1])
        if mode == "dev":
            best_test_f1 = self.model.best_dev_f1.eval()
            if f1 > best_test_f1:
                tf.assign(self.model.best_dev_f1, f1).eval()
                logger.info(">>> new best dev f1 score:{:>.3f}".format(f1))
                self.model.save_model(sess, FLAGS.ckpt_path, name="best_score.ckpt")
                logger.info(">>> best model saved. ")
            return f1 > best_test_f1
        elif mode == "test":
            best_test_f1 = self.model.best_test_f1.eval()
            if f1 > best_test_f1:
                tf.assign(self.model.best_test_f1, f1).eval()
                logger.info(">>> !!! Test f1 score:{:>.3f}".format(f1))
            return f1 > best_test_f1

    def evaluate_line(self):
        with tf.Session(config=tf_config) as sess:
            ### 1. create model and load parameters
            self.model = Model(self.config)
            ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_path)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                self.model.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                logger.info("Created model with fresh parameters.")
                sess.run(tf.global_variables_initializer())

            ### 2. Evaluating
            while True:
                try:
                    line = input("请输入测试句子:")
                    inputs = utils.input_from_line(line, self.ch_2_id)

                    trans = self.model.trans.eval(session=sess)
                    lengths, scores = self.model.run_step(sess, False, inputs)
                    batch_paths = self.model.decode(scores, lengths, trans)
                    tags = [self.id_2_tag[idx] for idx in batch_paths[0]]

                    result = utils.result_to_json(inputs[0][0], tags)
                    print(result)
                except Exception as e:
                    logger.info(e)


def main():
    runner = Runner()
    if FLAGS.train:
        if FLAGS.clean:
            utils.clean(FLAGS)
        runner.train()
    else:
        runner.evaluate_line()


if __name__ == "__main__":
    tf.app.run(main)