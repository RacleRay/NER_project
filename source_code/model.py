#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Author  :   Racle
@Version :   1.0
@Desc    :   None
'''
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers
from rnncell import CoupledInputForgetGateLSTMCell

import utils



class Model:
    def __init__(self, config):
        self.config = config

        self.lr = config["lr"]                # 学习率
        self.char_dim = config["char_dim"]    # word vec向量的维度
        self.lstm_dim = config["lstm_dim"]    # 选用LSTM模型时的隐层维度
        self.seg_dim = config["seg_dim"]      # seg标记(I,B,E,S)的向量表示维度
        self.embedding_dim = self.lstm_dim + self.seg_dim

        self.dilate_rate = config["dilate_rate"]     # [1, 1, 2]
        self.filter_width = config["filter_width"]   # 3
        self.num_filter = config["num_filter"]       # 100
        self.num_sublayers = config["num_sublayers"] # 4

        self.num_tags = config["num_tags"]    # named entity的种类数目
        self.num_chars = config["num_chars"]  # 样本中总字数
        self.num_segs = 4                     # seg标记的种类数

        self.model_type = config['model_type']# bilstm or idcnn

        self.dropout = tf.placeholder(dtype=tf.float32,
                                      name="Dropout")  # TFv1.x中为1-keepprob

        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)  # f1 score for dev set
        self.best_test_f1 = tf.Variable(0.0, trainable=False)

        self.initializer = initializers.xavier_initializer()

        # network
        self.__input_create()
        self.__model_create()

        # optimizer
        with tf.variable_scope("optimizer"):
            optimizer = self.config["optimizer"]
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.lr)
            else:
                raise KeyError
            # gradient clipping by value
            grad_vars = self.opt.compute_gradients(self.loss)
            clip_grad = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                         for g, v in grad_vars]
            self.train_op = self.opt.apply_gradients(clip_grad, self.global_step)

        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def __input_create(self):
        # add placeholders for the model
        self.char_inputs = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],
                                          name="ChatInputs")
        self.seg_inputs = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None],
                                         name="SegInputs")
        self.targets = tf.placeholder(dtype=tf.int32,
                                      shape=[None, None],
                                      name="Targets")
        # related params
        char_mark = tf.sign(tf.abs(self.char_inputs))  # 过滤0 -- padding标记
        actual_len = tf.reduce_sum(char_mark, reduction_indices=1)
        self.lengths = tf.cast(actual_len, tf.int32)
        self.batch_size = tf.shape(self.char_inputs)[0]
        self.num_steps = tf.shape(self.char_inputs)[-1] # 加上padding

    def __model_create(self):
        assert self.model_type.lower() in ['bilstm', 'idcnn'], "Model type must be 'bilstm' or 'idcnn'"
        # 1. embedding layer
        embedding = self.embedding_layer(self.char_input, self.seg_inputs)
        # 2. main network structure
        if self.model_type == 'bilstm':
            model_inputs = tf.nn.dropout(embedding, 1-self.dropout)
            model_outputs = self.biLSTM_layer(model_inputs)
            self.logits = self.project_layer_bilstm(model_outputs)
        else if self.model_type == 'idcnn':
            model_inputs = tf.nn.dropout(embedding, 1-self.dropout)
            model_outputs = self.IDCNN_layer(model_inputs)
            self.logits = self.project_layer_idcnn(model_outputs)
        # 3. loss compute layer
        self.loss = self.loss_layer(self.logits)

    def embedding_layer(self, char_inputs, seg_inputs, config, name="embedding"):
        """embedding重要且灵活，可以将词性、依存关系等特征进行编码处理，传入embedding
        此处没有使用预训练的词向量进行初始化。在Runner的main函数中进行embedding assign。
        :param char_inputs: indexes of sentence
        :param seg_inputs: segmentation feature, length of each word
        :param config: segmentation feature dim
        :return: [1, num_steps, embedding size]

        e.g. 高:3 血:22 糖:23 和:24 高:3 血:22 压:25 => char_inputs=[3,22,23,24,3,22,25]
             seg_inputs 高血糖=[1,2,3] 和=[0] 高血压=[1,2,3] => seg_inputs=[1,2,3,0,1,2,3]
        """
        embedding = []
        with tf.variable_scope(name), tf.device('/cpu:0'):
            self.char_lookup = tf.get_variable(
                    name="char_embedding",
                    shape=[self.num_chars, self.char_dim],
                    initializer=self.initializer)
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))

            if config["seg_dim"]:
                with tf.variable_scope("seg_embedding"), tf.device('/cpu:0'):
                    self.seg_lookup = tf.get_variable(
                        name="seg_embedding",
                        shape=[self.num_segs, self.seg_dim],
                        initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))

            embed = tf.concat(embedding, axis=-1)
        return embed

    def biLSTM_layer(self, lstm_inputs, name="BiLSTM_layer"):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, 2*lstm_dim]
        """
        with tf.variable_scope(name):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = CoupledInputForgetGateLSTMCell(
                        self.lstm_dim,
                        use_peepholes=True,
                        initializer=self.initializer)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                lstm_inputs,
                dtype=tf.float32,
                sequence_length=self.lengths)
        return tf.concat(outputs, axis=2)

    def project_layer_bilstm(self, lstm_outputs, name="fc_layer_bilstm"):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope(name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.lstm_dim*2, self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)
                b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim*2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))
            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)
                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                pred = tf.nn.xw_plus_b(hidden, W, b)
            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])

    def IDCNN_layer(self, model_inputs, name="IDCNN_layer"):
        """IDCNN layer

        shape of input after expand = [batch, in_height, in_width, in_channels]
            in_height：一句话为1，in_width：seq length，in_channels：embedding dim
        shape of filter = [filter_height, filter_width, in_channels, out_channels]
            in_channels：embedding dim，out_channels：number of filters

        关于 tf.nn.atrous_conv2d(value,filters,rate,padding,name=None）
            value： [batch, height, width, channels]这样的shape，
                [batch_size, sentence高度(1), sentence宽度(length), sentence通道数(embedding dim)]
            filters：[filter_height, filter_width, channels, out_channels]
                [卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
            rate：int型的正数，空洞卷积默认stride=1
                rate参数，它定义为我们在输入图像上卷积时的采样间隔，相当于卷积核中穿插了（rate-1）数量的“0”，
                这样做卷积时就相当于对原图像的采样间隔变大了。rate=1时，就没有0插入，相当于普通卷积。
            padding： string，”SAME”,”VALID”其中之一，决定不同边缘填充方式。
                “VALID”，返回[batch,height-(filter_height + (filter_height - 1) * (rate - 1))+1,
                    width-(filter_width + (filter_width - 1) * (rate - 1))+1,out_channels]的Tensor
                “SAME”，返回[batch, height, width, out_channels]的Tensor

        :param model_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, totalChannels]
        """
        model_inputs = tf.expand_dims(model_inputs, 1)  # 化为图像相同维度进行处理
        with tf.variable_scope(name):
            # init filter weights
            shape = [1, self.filter_width, self.embedding_dim, self.num_filter]
            weights = tf.get_variable("idcnn_filter",
                                      shape=shape,
                                      initializer=self.initializer)
            # first cnn layer
            cnn_out = tf.nn.conv2d(model_inputs,
                                   weights,
                                   strides=[1,1,1,1],
                                   padding="SAME",
                                   use_cudnn_on_gpu=True,
                                   name="first cnn layer")
            # dilate cnn layers
            reuseFlag = True if self.dropout == 0.0 else False
            eachModuleOut = []
            totalChannels = 0
            for i in range(self.num_sublayers):  # dilate cnn modules
                for j in range(len(self.dilate_rate)):  # dilate layers of one module
                    # reuse first layer, or when dropout rate is 0.0
                    with tf.variable_scope("atrous-conv-layer-%d" % i,
                                           reuse=True if (reuseFlag or i > 0) else False):
                        w = tf.get_variable(
                              "weights",
                              shape=[1, self.filter_width, self.num_filter, self.num_filter],
                              initializer=tf.contrib.layers.xavier_initializer())
                        b = tf.get_variable("bias", shape=[self.num_filter])
                        atrous_conv_out = tf.nn.atrous_conv2d(cnn_out,
                                                              w,
                                                              rate=self.dilate_rate[j],
                                                              padding="SAME")
                        conv = tf.nn.bias_add(atrous_conv_out, b)
                        conv = tf.nn.relu(conv)
                        # record every sub modules` outputs
                        if j == (len(self.dilate_rate) - 1):
                            eachModuleOut.append(conv)
                        # iterate cnn inputs
                        cnn_out = conv
            totalChannels = self.num_filter * self.num_sublayers
            # aggregate outputs
            cnn_outs = tf.concat(values=eachModuleOut, axis=3)
            cnn_outs = tf.dropout(cnn_outs, 1 - self.dropout)
            cnn_outs = tf.squeeze(cnn_outs, [1])  # expanded dim
            cnn_outs = tf.reshape(cnn_outs, [-1, totalChannels])
            self.cnn_out_channels = totalChannels
            return cnn_outs

    def project_layer_idcnn(self, idcnn_outputs, name="fc_layer_idcnn"):
        """
        Project layer for idcnn
        :param idcnn_outputs: [batch_size, num_steps, cnn_out_channels]
        :return: [batch_size, num_steps, num_tags]
        """
         with tf.variable_scope(name):
            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.cnn_out_channels, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)
                b = tf.get_variable("b",  initializer=tf.constant(0.001, shape=[self.num_tags]))
                pred = tf.nn.xw_plus_b(idcnn_outputs, W, b)
            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])

    def loss_layer(self, project_logits, name="crf_loss"):
        """
        calculate crf loss
        :param project_logits: [batch_size, num_steps, num_tags]
        :return: scalar loss
        """
        small_value = -1000.0
        with tf.variable_scope(name):
            # 1. add "start" sign in the state transitions sequence
            vertical_pad = tf.cast(tf.ones([self.batch_size, self.num_steps, 1]) * small_value,
                                   tf.float32)
            horizontal_pad = tf.concat(
                tf.ones(shape=[self.batch_size, 1, self.num_tags]) * small_value,
                tf.zeros(shape=[self.batch_size, 1, 1]),
                axis=-1)  # zero: start 不会转移到 start
            logits = tf.concat([project_logits, horizontal_pad], axis=1)
            logits = tf.concat([vertical_pad, project_logits], axis=-1)
            # pad targets too
            targets = tf.concat(
                [tf.cast(self.num_tags * tf.ones([self.batch_size, 1]), tf.int32), self.targets],
                axis=-1)  # self.num_tags: the index of the new "start" sign

            # 2. transmition params to learn
            self.trans = tf.get_variable("transitions",
                                         shape=[self.num_tags + 1, self.num_tags + 1],
                                         initializer=self.initializer)

            # 3. compute the log likelihood for defining loss of the whole model
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=self.lengths+1)
            return tf.reduce_mean(-log_likelihood)

    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        # 1. feed data
        _, chars, segs, tags = batch
        feed_dict = {
            self.char_inputs: np.asarray(chars),
            self.seg_inputs: np.asarray(segs),
            self.dropout: 0.0,
        }
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.config["dropout"]
        # 2. run
        if is_train:
            global_step, loss, _ = sess.run(
                [self.global_step, self.loss, self.train_op],
                feed_dict)
            return global_step, loss
        else:
            lengths, logits = sess.run([self.lengths, self.logits],
                                       feed_dict)
            return lengths, logits

    def evaluate(self, sess, data_manager, id_to_tag):
        """
        :param sess: session to run the model
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result  ->  格式 "char 正确tag 预测tag"
        """
        results = []
        trans_matrix = self.trans.eval()
        for batch in data_manager.iter_batch():
            chars = batch[0]
            tag_idx = batch[-1]
            lengths, scores = self.run_step(sess, is_train=False, batch)
            batch_paths = self.decode(scores, lengths, trans_matrix)
            for i in range(len(chars)):  # len(chars) -> batch size
                result = []
                string = chars[i][:lengths[i]]  # 有效字符
                gold = utils.iobes_iob([id_to_tag[int(x)] for x in tag_idx[i][: lengths[i]]])
                pred = utils.iobes_iob([id_to_tag[int(x)] for x in batch_paths[i][: lengths[i]]])
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        return results

    @staticmethod
    def decode(logits, lengths, matrix):
        """
        inference final labels usa viterbi Algorithm
        :param logits: [batch_size, num_steps, num_tags]
        :param lengths: [batch_size]  # real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        paths = []
        small_value = -1000.0
        start_state = np.asarray([[small_value] * self.num_tags + [0]])
        for score, length in zip(logits, lengths):
            score = score[: length]
            pad = small_value * np.ones([length, 1])
            logits_padded = np.concatenate([score, pad], axis=1)
            logits_padded = np.concatenate([start_state, logits_padded], axis=0)
            path, _ = viterbi_decode(logits_padded, matrix)
            paths.append(path[1: ])
        return paths

    def save_model(self, sess, path, name="ner.ckpt"):
        checkpoint_path = os.path.join(path, name)
        self.saver.save(sess, checkpoint_path)