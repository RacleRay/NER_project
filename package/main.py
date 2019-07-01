#-*-encoding=utf8-*-
import json
import codecs
import itertools
from collections import OrderedDict
import os
import sys
from itertools import chain
import tensorflow as tf
import numpy as np
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers
currentPath=os.getcwd()
sys.path.append(currentPath)
import jieba
import jieba.posseg as pseg
root_path=os.getcwd()
global pyversion
if sys.version>'3':
    pyversion='three'
else:
    pyversion='two'
if pyversion=='three':
    import pickle
else :
    import cPickle,pickle
root_path=os.getcwd()+os.sep

CONFIG= {}

class Model(object):
    """模型结构类"""
    def __init__(self, config):

        self.config = config
        
        self.lr = config["lr"]
        self.char_dim = config["char_dim"]
        self.lstm_dim = config["lstm_dim"]
        self.seg_dim = config["seg_dim"]

        self.num_tags = config["num_tags"]
        self.num_chars = config["num_chars"]#样本中总字数
        self.num_segs = 4
        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
        self.initializer = initializers.xavier_initializer()
        self.char_inputs = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],
                                          name="ChatInputs")
        self.seg_inputs = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None],
                                         name="SegInputs")

        self.targets = tf.placeholder(dtype=tf.int32,
                                      shape=[None, None],
                                      name="Targets")
        # dropout keep prob
        self.dropout = tf.placeholder(dtype=tf.float32,
                                      name="Dropout")

        used = tf.sign(tf.abs(self.char_inputs))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.char_inputs)[0]
        self.num_steps = tf.shape(self.char_inputs)[-1]
        self.model_type = config['model_type']
        self.layers = [
            {
                'dilation': 1
            },
            {
                'dilation': 1
            },
            {
                'dilation': 2
            },
        ]
        self.filter_width = 3
        self.num_filter = self.lstm_dim 
        self.embedding_dim = self.char_dim + self.seg_dim
        self.repeat_times = 4
        self.cnn_output_width = 0
        embedding = self.embedding_layer(self.char_inputs, self.seg_inputs, config)
        if self.model_type == 'idcnn':
            model_inputs = tf.nn.dropout(embedding, self.dropout)
            model_outputs = self.IDCNN_layer(model_inputs)
            self.logits = self.project_layer_idcnn(model_outputs)
        
        else:
            raise KeyError

        self.loss = self.loss_layer(self.logits, self.lengths)

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
            grads_vars = self.opt.compute_gradients(self.loss)
            capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                 for g, v in grads_vars]
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def embedding_layer(self, char_inputs, seg_inputs, config, name=None):

        embedding = []
        self.char_inputs_test=char_inputs
        self.seg_inputs_test=seg_inputs
        with tf.variable_scope("char_embedding" if not name else name):
            self.char_lookup = tf.get_variable(
                    name="char_embedding",
                    shape=[self.num_chars, self.char_dim],
                    initializer=self.initializer)
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
            if config["seg_dim"]:
                with tf.variable_scope("seg_embedding"):
                    self.seg_lookup = tf.get_variable(
                        name="seg_embedding",
                        #shape=[4*20]
                        shape=[self.num_segs, self.seg_dim],
                        initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))
            embed = tf.concat(embedding, axis=-1)
        self.embed_test=embed
        self.embedding_test=embedding
        return embed

    def IDCNN_layer(self, model_inputs, 
                    name=None):
        model_inputs = tf.expand_dims(model_inputs, 1)
        self.model_inputs_test=model_inputs
        reuse = False
        if self.dropout == 1.0:
            reuse = True
        with tf.variable_scope("idcnn" if not name else name):
            #shape=[1*3*120*100]
            shape=[1, self.filter_width, self.embedding_dim,
                       self.num_filter]
            print(shape)
            filter_weights = tf.get_variable(
                "idcnn_filter",
                shape=[1, self.filter_width, self.embedding_dim,
                       self.num_filter],
                initializer=self.initializer)
            layerInput = tf.nn.conv2d(model_inputs,
                                      filter_weights,
                                      strides=[1, 1, 1, 1],
                                      padding="SAME",
                                      name="init_layer",use_cudnn_on_gpu=True)
            self.layerInput_test=layerInput
            finalOutFromLayers = []
            
            totalWidthForLastDim = 0
            for j in range(self.repeat_times):
                for i in range(len(self.layers)):
                    #1,1,2
                    dilation = self.layers[i]['dilation']
                    isLast = True if i == (len(self.layers) - 1) else False
                    with tf.variable_scope("atrous-conv-layer-%d" % i,
                                           reuse=True
                                           if (reuse or j > 0) else False):
                        w = tf.get_variable(
                            "filterW",
                            shape=[1, self.filter_width, self.num_filter,
                                   self.num_filter],
                            initializer=tf.contrib.layers.xavier_initializer())
                        if j==1 and i==1:
                            self.w_test_1=w
                        if j==2 and i==1:
                            self.w_test_2=w                            
                        b = tf.get_variable("filterB", shape=[self.num_filter])
                        conv = tf.nn.atrous_conv2d(layerInput,
                                                   w,
                                                   rate=dilation,
                                                   padding="SAME")
                        self.conv_test=conv 
                        conv = tf.nn.bias_add(conv, b)
                        conv = tf.nn.relu(conv)
                        if isLast:
                            finalOutFromLayers.append(conv)
                            totalWidthForLastDim += self.num_filter
                        layerInput = conv
            finalOut = tf.concat(axis=3, values=finalOutFromLayers)
            keepProb = 1.0 if reuse else 0.5
            finalOut = tf.nn.dropout(finalOut, keepProb)
            finalOut = tf.squeeze(finalOut, [1])
            finalOut = tf.reshape(finalOut, [-1, totalWidthForLastDim])
            self.cnn_output_width = totalWidthForLastDim
            return finalOut

    def project_layer_idcnn(self, idcnn_outputs, name=None):

        with tf.variable_scope("project"  if not name else name):
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.cnn_output_width, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b",  initializer=tf.constant(0.001, shape=[self.num_tags]))

                pred = tf.nn.xw_plus_b(idcnn_outputs, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])

    def loss_layer(self, project_logits, lengths, name='crf_loss'):  
        """条件随机场损失计算"""
        with tf.variable_scope(name):
            small = -1000.0
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
            logits = tf.concat([project_logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)
            targets = tf.concat(
                [tf.cast(self.num_tags*tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)

            self.trans = tf.get_variable(
                "transitions",
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=self.initializer)              
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lengths+1)
            return tf.reduce_mean(-log_likelihood)

    def create_feed_dict(self, batch):
        """
        batch包括
        chars: id list
        segs: 词长度标记list  e.g. 0，1，3，1，2，3
        tags：空list
        """
        _, chars, segs, tags = batch
        feed_dict = {
            self.char_inputs: np.asarray(chars),
            self.seg_inputs: np.asarray(segs),
            self.dropout: 1.0,  # 预测阶段
        }
        return feed_dict

    def run_step(self, sess, batch):
        """
        :return: lengths 序列长度； logits 预测的各标记概率
        """
        feed_dict = self.create_feed_dict(batch)
        lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
        return lengths, logits

    def decode(self, logits, lengths, matrix):
        """
        解码器，使用crf转移矩阵和idcnn预测的概率分布，通过viterbi算法求解
        :param logits: idcnn预测的概率分布
        :param lengths: 每个batch的文本长度，list
        :param matrix: crf转移矩阵
        :return: paths - [seq_len] list of integers containing the highest scoring tag indices
        """
        paths = []
        small = -1000.0
        start = np.asarray([[small]*self.num_tags + [0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            # 右侧加一列1，上侧加一行-1000
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)

            # logits：[seq_len, num_tags] matrix of unary potentials
            # matrix：[num_tags, num_tags] matrix of binary potentials
            path, _ = viterbi_decode(logits, matrix)

            paths.append(path[1:])
        return paths

    def result_to_json(self,string, tags):
        """
        计算结果处理为json文件
        :param string: 文本序列
        :param tags: 标记序列，list
        :return: dict：{"string": string, "entities": []}
        """
        item = {"string": string, "entities": []}
        entity_name = ""
        entity_start = 0
        idx = 0
        # 按标记输出ner结果
        for char, tag in zip(string, tags):
            if tag[0] == "S":  # single
                item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":tag[2:]})
            elif tag[0] == "B":  # begin
                entity_name += char
                entity_start = idx
            elif tag[0] == "I":  # in
                entity_name += char
            elif tag[0] == "E":  # end
                entity_name += char
                item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2:]})
                entity_name = ""
            else: # O, 非ner识别目标
                entity_name = ""
                entity_start = idx
            idx += 1
        return item

    def evaluate_line(self, sess, inputs, id_to_tag):
        """
        计算主要过程
        :param sess: 计算图
        :param inputs: list, [0]维-》文本，[1]维-》id，[2]维-》词长度标记，[3]维-》空list
        :param id_to_tag: dict，标记id到tag名称的map
        :return: json，实体标记结果
        """
        # trans标记与标记之间的转移矩阵
        trans = self.trans.eval(session=sess)
        # 计算标记分类概率scores
        lengths, scores = self.run_step(sess, inputs)
        # 解码计算出各个词的标记id
        batch_paths = self.decode(scores, lengths, trans)
        # 标记id转为标记
        tags = [id_to_tag[idx] for idx in batch_paths[0]]
        return self.result_to_json(inputs[0][0], tags)


class Chunk(object):
    """模型计算类"""
    def __init__(self): 
        self.config_file = json.load(open("config_file", encoding="utf8"))
        self.tf_config = tf.ConfigProto()
        self.sess = tf.Session(config=self.tf_config)
        self.sess.run(tf.global_variables_initializer()) 
        self.maps = "maps.pkl"
        # 读取字和index，标记和index的映射关系
        if pyversion=='three':
            self.char_to_id, self.id_to_char, self.tag_to_id, self.id_to_tag = pickle.load(open(self.maps, "rb"))
        else:
            self.char_to_id, self.id_to_char, self.tag_to_id, self.id_to_tag = pickle.load(open(self.maps, "rb"), protocol=2)
        self.model = Model(self.config_file)
        # 读取预训练模型
        self.ckpt = tf.train.get_checkpoint_state("ckpt")
        if self.ckpt and tf.train.checkpoint_exists(self.ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % self.ckpt.model_checkpoint_path)
            self.model.saver.restore(self.sess, self.ckpt.model_checkpoint_path)
        else:
            print("No model file")
        
    def features(self,string):
        """处理词组长度特征：0为单个字，1为词开头，3为词结尾
        len(word) = 1, 特征为0
        len(word) = 2, 特征为1，3
        len(word) > 2，特征为1，2，...，3
        """
        def _w2f(word):
            lenth=len(word)
            if lenth==1:
                r=[0]
            if lenth>1:
                r=[2]*lenth
                r[0]=1
                r[-1]=3
            return r
        return list(chain.from_iterable([_w2f(word) for word in jieba.cut(string) if len(word.strip())>0]))
      
        
    def get_text_input(self,text):
        """处理输入文本，创建输入格式的数据"""
        inputs = list()
        inputs.append([text])   # 原文本list特征
        D = self.char_to_id["<UNK>"]  # D = 1
         
        inputs.append([[self.char_to_id.setdefault(char, D) 
                            for char in text if len(char.strip())>0]])  # 字转id的id list
        inputs.append([self.features(text)])  # 词长度特征 list
        inputs.append([[]])  # 空 list

        # 计算输出
        if len(text.strip())>1:
            return self.model.evaluate_line(self.sess, inputs, self.id_to_tag)
        
if __name__ == "__main__":
    # 在设计过程中，减少重复，初始化操作控制在一次。高效，list comprehension的实现，不会多次向系统申请较大的内存
    c = Chunk()
    for line in open('text.txt','r',encoding='utf8'):
        print(c.get_text_input(line.strip()))

  

