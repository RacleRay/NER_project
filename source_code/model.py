# encoding = utf-8
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers

from utils import result_to_json
from data_utils import create_input, iobes_iob, iob_iobes


class Model(object):
    """
    model类
    """
    def __init__(self, config):

        self.config = config
        
        self.lr = config["lr"]
        self.char_dim = config["char_dim"]
        self.lstm_dim = config["lstm_dim"]
        self.seg_dim = config["seg_dim"]

        self.num_tags = config["num_tags"]
        self.num_chars = config["num_chars"]  # 样本中总字数
        self.num_segs = 4

        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
        self.initializer = initializers.xavier_initializer()

        # add placeholders for the model
        # word vector
        self.char_inputs = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],
                                          name="ChatInputs")
        # {0,1,2,3}的vector表示
        self.seg_inputs = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None],
                                         name="SegInputs")
        # tag 
        self.targets = tf.placeholder(dtype=tf.int32,
                                      shape=[None, None],
                                      name="Targets")
        self.dropout = tf.placeholder(dtype=tf.float32,
                                      name="Dropout")

        used = tf.sign(tf.abs(self.char_inputs))  # padding为0，used=0
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.char_inputs)[0]
        self.num_steps = tf.shape(self.char_inputs)[-1]

        #Add model type by crownpku bilstm or idcnn：idcnn is better
        self.model_type = config['model_type']
        #parameters for idcnn: dilate rate
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
        # char和seg，concat为embedding表示
        self.embedding_dim = self.char_dim + self.seg_dim
        self.repeat_times = 4
        self.cnn_output_width = 0
        
        # embeddings for chinese character and segmentation representation
        embedding = self.embedding_layer(self.char_inputs, self.seg_inputs, config)

        if self.model_type == 'bilstm':
            # apply dropout before feed to lstm layer
            model_inputs = tf.nn.dropout(embedding, self.dropout)

            # bi-directional lstm layer
            model_outputs = self.biLSTM_layer(model_inputs, self.lstm_dim, self.lengths)

            # logits for tags
            self.logits = self.project_layer_bilstm(model_outputs)
        
        elif self.model_type == 'idcnn':
            # apply dropout before feed to idcnn layer
            model_inputs = tf.nn.dropout(embedding, self.dropout)

            # ldcnn layer
            model_outputs = self.IDCNN_layer(model_inputs)

            # logits for tags
            self.logits = self.project_layer_idcnn(model_outputs)
        
        else:
            raise KeyError

        # loss of the model   you can read these code from loss
        self.loss = self.loss_layer(self.logits, self.lengths)

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

            # apply grad clip to avoid gradient explosion
            grads_vars = self.opt.compute_gradients(self.loss)
            capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                 for g, v in grads_vars]
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)

        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)


    def embedding_layer(self, char_inputs, seg_inputs, config, name=None):
        """
        embedding重要且灵活，甚至可以将词性、依存关系等特征进行编码处理，传入embedding
        :param char_inputs: indexes of sentence
        :param seg_inputs: segmentation feature, length of each word
        :param config: segmentation feature dim
        :return: [1, num_steps, embedding size]
        """
        # 高:3 血:22 糖:23 和:24 高:3 血:22 压:25 char_inputs=[3,22,23,24,3,22,25]
        # ‘高血糖’ ‘和’ ‘高血压’ seg_inputs 高血糖=[1,2,3] 和=[0] 高血压=[1,2,3]  seg_inputs=[1,2,3,0,1,2,3]
        embedding = []
        self.char_inputs_test=char_inputs
        self.seg_inputs_test=seg_inputs
        with tf.variable_scope("char_embedding" if not name else name), tf.device('/gpu:0'):

            # 输入char_inputs='常' 对应的字典的索引为：8
            # self.char_lookup是2677*100的tensor，char_inputs字对应在字典的索引
            self.char_lookup = tf.get_variable(
                    name="char_embedding",
                    shape=[self.num_chars, self.char_dim],
                    initializer=self.initializer)
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))

            if config["seg_dim"]:
                with tf.variable_scope("seg_embedding"), tf.device('/gpu:0'):
                    self.seg_lookup = tf.get_variable(
                        name="seg_embedding",
                        shape=[self.num_segs, self.seg_dim],  # 4*20的tensor
                        initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))

            embed = tf.concat(embedding, axis=-1)
        self.embed_test=embed
        self.embedding_test=embedding
        return embed


    def biLSTM_layer(self, model_inputs, lstm_dim, lengths, name=None):
            """
            :param lstm_inputs: [batch_size, num_steps, emb_size] 
            :return: [batch_size, num_steps, 2*lstm_dim] 
            """
            with tf.variable_scope("char_BiLSTM" if not name else name):
                lstm_cell = {}
                for direction in ["forward", "backward"]:
                    with tf.variable_scope(direction):
                        lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                            lstm_dim,
                            use_peepholes=True,
                            initializer=self.initializer,
                            state_is_tuple=True)
                outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                    lstm_cell["forward"],
                    lstm_cell["backward"],
                    model_inputs,
                    dtype=tf.float32,
                    sequence_length=lengths)
            return tf.concat(outputs, axis=2)


    def IDCNN_layer(self, model_inputs, 
                    name=None):
        """IDCNN layer
        :param model_inputs: [batch_size, num_steps, emb_size]
        # emb_size在IDCNN中成了channel数量，这点很重要
        :return: [batch_size, num_steps, cnn_output_width]
        """
        # tf.expand_dims会向tensor中插入一个维度，插入位置就是参数代表的位置（维度从0开始）
        model_inputs = tf.expand_dims(model_inputs, 1)
        self.model_inputs_test=model_inputs
        reuse = False

        if self.dropout == 1.0:
            reuse = True
        with tf.variable_scope("idcnn" if not name else name):
            shape=[1, self.filter_width, self.embedding_dim,
                   self.num_filter]   # [1*3*120*100]
            print(shape)
            filter_weights = tf.get_variable(
                "idcnn_filter",
                shape=shape,
                initializer=self.initializer)

            # shape of input = [batch, in_height, in_width, in_channels]
            # in_height：一句话为1，in_width：seq length，in_channels：embedding dim
            # shape of filter = [filter_height, filter_width, in_channels, out_channels]
            # in_channels：embedding dim，out_channels：number of filters

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
                    dilation = self.layers[i]['dilation']  # 1,1,2
                    isLast = True if i == (len(self.layers) - 1) else False

                    with tf.variable_scope("atrous-conv-layer-%d" % i,
                                           reuse=True
                                           if (reuse or j > 0) else False):
                        # shape: 卷积核的高度，卷积核的宽度，通道数，卷积核个数
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

                '''tf.nn.atrous_conv2d(value,filters,rate,padding,name=None）
                    # value： [batch, height, width, channels]这样的shape，
                        # [训练时一个batch的sentence数量, sentence高度(1), sentence宽度(length), sentence通道数(embedding dim)]
                    # filters：[filter_height, filter_width, channels, out_channels]
                        # [卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
                    # rate：int型的正数，空洞卷积默认stride=1，也就是滑动步长无法改变，固定为1
                        # rate参数，它定义为我们在输入图像上卷积时的采样间隔，相当于卷积核中穿插了（rate-1）数量的“0”，
                        # 这样做卷积时就相当于对原图像的采样间隔变大了。rate=1时，就没有0插入，相当于普通卷积。
                    # padding： string，”SAME”,”VALID”其中之一，决定不同边缘填充方式。
                        # “VALID”，返回[batch,height-(filter_height + (filter_height - 1) * (rate - 1))+1,
                                    # width-(filter_width + (filter_width - 1) * (rate - 1))+1,out_channels]的Tensor
                        # “SAME”，返回[batch, height, width, out_channels]的Tensor'''
                        conv = tf.nn.atrous_conv2d(layerInput,
                                                   w,
                                                   rate=dilation,
                                                   padding="SAME")
                        self.conv_test = conv
                        conv = tf.nn.bias_add(conv, b)
                        conv = tf.nn.relu(conv)
                        if isLast:  # 计算repeat_times次
                            finalOutFromLayers.append(conv)
                            totalWidthForLastDim += self.num_filter
                        layerInput = conv
            finalOut = tf.concat(axis=3, values=finalOutFromLayers)
            keepProb = 1.0 if reuse else 0.5
            finalOut = tf.nn.dropout(finalOut, keepProb)

            # tf.squeeze
            # 给定张量输入，此操作返回相同类型的张量，并删除所有尺寸为1的尺寸。
            # 如果不想删除所有尺寸1尺寸，可以通过指定axis来删除特定尺寸1尺寸。
            # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
            # tf.shape(tf.squeeze(t, [2, 4]))  # [1, 2, 3, 1]
            finalOut = tf.squeeze(finalOut, [1])
            finalOut = tf.reshape(finalOut, [-1, totalWidthForLastDim])
            self.cnn_output_width = totalWidthForLastDim
            return finalOut


    def project_layer_bilstm(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"  if not name else name):
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
    

    def project_layer_idcnn(self, idcnn_outputs, name=None):
        """
        Project layer for idcnn by crownpku
        :param lstm_outputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"  if not name else name):
            
            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.cnn_output_width, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b",  initializer=tf.constant(0.001, shape=[self.num_tags]))

                pred = tf.nn.xw_plus_b(idcnn_outputs, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])


    def loss_layer(self, project_logits, lengths, name=None):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"  if not name else name):
            small = -1000.0
            # pad logits for crf loss
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

            # crf_log_likelihood在一个条件随机场里面计算标签序列的log-likelihood
                # inputs: 一个形状为[batch_size, max_seq_len, num_tags] 的tensor,
                # tag_indices: 一个形状为[batch_size, max_seq_len] 的矩阵,其实就是真实标签.
                # sequence_lengths: 一个形状为 [batch_size] 的向量,表示每个序列的长度.
                # transition_params: 形状为[num_tags, num_tags] 的转移矩阵
                # log_likelihood: 标量
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lengths+1)
            return tf.reduce_mean(-log_likelihood)


    def create_feed_dict(self, is_train, batch):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data 
        :return: structured data to feed
        """
        _, chars, segs, tags = batch
        feed_dict = {
            self.char_inputs: np.asarray(chars),
            self.seg_inputs: np.asarray(segs),
            self.dropout: 1.0,
        }
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.config["dropout_keep"]
        return feed_dict


    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss, _ = sess.run(
                [self.global_step, self.loss, self.train_op],
                feed_dict)
            # for debug
            # global_step, loss,_,char_lookup_out,seg_lookup_out,char_inputs_test,seg_inputs_test,embed_test,embedding_test,\
            #     model_inputs_test,layerInput_test,conv_test,w_test_1,w_test_2,char_inputs_test= sess.run(
            #     [self.global_step, self.loss, self.train_op,self.char_lookup,self.seg_lookup,self.char_inputs_test,self.seg_inputs_test,\
            #      self.embed_test,self.embedding_test,self.model_inputs_test,self.layerInput_test,self.conv_test,self.w_test_1,self.w_test_2,self.char_inputs],
            #     feed_dict)
            return global_step, loss
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
            return lengths, logits


    def decode(self, logits, lengths, matrix):
        """
        inference final labels usa viterbi Algorithm
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        paths = []
        small = -1000.0
        start = np.asarray([[small]*self.num_tags +[0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)

            paths.append(path[1:])
        return paths


    def evaluate(self, sess, data_manager, id_to_tag):
        """
        :param sess: session  to run the model 
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        trans = self.trans.eval()
        for batch in data_manager.iter_batch():
            strings = batch[0]
            tags = batch[-1]
            lengths, scores = self.run_step(sess, False, batch)
            batch_paths = self.decode(scores, lengths, trans)
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                gold = iobes_iob([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
                pred = iobes_iob([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])

                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        return results


    def evaluate_line(self, sess, inputs, id_to_tag):
        """
        单行预测
        """
        trans = self.trans.eval(session=sess)
        lengths, scores = self.run_step(sess, False, inputs)
        batch_paths = self.decode(scores, lengths, trans)
        tags = [id_to_tag[idx] for idx in batch_paths[0]]
        return result_to_json(inputs[0][0], tags)
