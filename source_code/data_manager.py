#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_manager.py
@Author  :   Racle
@Version :   1.0
@Desc    :   None
'''

import os
import re
import codecs
import random
import jieba
import utils

################ load data #################
def load_sentences(path, digits_to_zeros):
    "输入文件每一行为一个字+它的标记，或者是空行表示下一个输入文本"
    sentences = []
    sentence = []
    for line in open(path, 'r',encoding='utf8'):
        line = zero_digits(line.rstrip()) if digits_to_zeros else line.rstrip()
        if not line:  # 空行下一个输入
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
        else:
            if line[0] == " ":
                line = "$" + line[1:]
                word = line.split()
            else:
                word= line.split( )
            assert len(word) == 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


def update_tag_scheme(sentences, tag_scheme):
    "tag_scheme: 'iob' or 'iobes'"
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # iob2 function:
        #   1. Check that tags are given in the IOB format
        #   2. Modify error tagging
        if not utils.iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme.lower() == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme.lower() == 'iobes':
            new_tags = utils.iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


################ create id and item mapping #################
def char_mapping(sentences, do_lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    chars = [[x[0].lower() if do_lower else x[0] for x in s] for s in sentences]
    dic = create_dic(chars)
    dic["<PAD>"] = 99999999  # 选取较大的假设频率数值
    dic['<UNK>'] = 99999999  # 保证出现在字典中
    char_to_id, id_to_char = create_mapping(dic)
    return dic, char_to_id, id_to_char


def create_dic(words_of_sentences):
    dic = {}
    for words in words_of_sentences:
        for w in words:
            count = dic.get(w, 0)
            dic[w] = count + 1
    return dic


def create_mapping(dic):
    "按频率进行排序"
    sorted_list = sorted(dic.items(), key=lambda x: (-x[1], x[0]))
    id_2_item = {i: v[0] for i, v in enumerate(sorted_list)}
    item_2_id = {v: k for k, v in id_2_item.items()}
    return item_2_id, id_2_item


def augment_with_pretrained(dic_of_trainset, ext_emb_path, chars):
    """通过有pretrained embedding的字来扩展训练用的dic，有两种方式：
    1. 给出"chars", 将通过"chars"筛选字，只选择存在于"chars"中的；
    2. "chars"为None，不对pretrained embedding的字进行筛选，直接全部加入。
    """
    print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])

    if chars is None:
        for char in pretrained:
            if char not in dic_of_trainset:
                dic_of_trainset[char] = 0  # 扩展词，取小的频率
    else:
        for char in chars:
            if any(x in pretrained for x in [
                char,
                char.lower(),
                re.sub('\d', '0', char.lower())
            ]) and char not in dic_of_trainset:
                dic_of_trainset[char] = 0

    char_to_id, id_to_char = create_mapping(dic_of_trainset)
    return dic_of_trainset, char_to_id, id_to_char


def tag_mapping(sentences, tagId_path, idTag_path):
    ti = open(tagId_path, 'w', encoding='utf-8')
    it = open(idTag_path, 'w', encoding='utf-8')
    # tags of sentences
    tags = []
    for s in sentences:
        ts = []
        for char in s:
            tag = char[-1]
            ts.append(tag)
        tags.append(ts)
    # create mapping
    dic = create_dic(tags)
    tag_to_id, id_to_tag = create_mapping(dic)
    # save
    for k,v in tag_to_id.items():
        ti.write(k + ":" + str(v) + "\n")
    for k,v in id_to_tag.items():
        it.write(str(k) + ":" + str(v) + "\n")

    return dic, tag_to_id, id_to_tag


################ data manager #################
def prepare_dataset(sentences, char_to_id, tag_to_id, lower=False, train=True):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - [0]text: cut by char
        - [1]char index  -- 字特征
        - [2]seg index   -- bies特征
        - [3]tag index   -- ground truth
    """
    none_index = tag_to_id["O"]

    def f(x):
        return x.lower() if lower else x

    data = []
    for s in sentences:
        chars = [w[0] for w in s]
        char_idx = [char_to_id[f(w) if f(w) in char_to_id else '<UNK>']
                    for w in chars]
        seg_idx = get_seg_features("".join(chars))
        if train:
            tag_idx = [tag_to_id[w[-1]] for w in s]
        else:
            tag_idx = [none_index for _ in chars]
        data.append([chars, char_idx, seg_idx, tag_idx])

    return data


def get_seg_features(string):
    """
    Segment text with jieba
    features are represented in bies format:
    每个词：
        single   -- 编码 -- 0
        begin    -- 编码 -- 1
        interior -- 编码 -- 2
        end      -- 编码 -- 3
    所有词，包括非entity都进行
    """
    seg_feature = []
    for word in jieba.cut(string):
        if len(word) == 1:
            seg_feature.append(0)
        else:
            tmp = [2] * len(word)
            tmp[0] = 1
            tmp[-1] = 3
            seg_feature.extend(tmp)
    return seg_feature


class BatchManager:
    def __init__(self, sentences, batch_size, ch2id, tag2id, do_lower=True, train_mode=True):
        data = prepare_dataset(sentences, ch2id, tag2id, do_lower, train_mode)
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.num_batches = len(self.batch_data)
        self.num_data = len(data)

    def sort_and_pad(self, data, batch_size):
        num_batch = len(data) // batch_size
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = []
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data))
        return batch_data

    @staticmethod
    def pad_data(data):
        chars, char_idx, seg_idx, tag_idx = [], [], [], []
        max_len = max([len(sent[0]) for sent in data])
        for line in data:
            ch, ci, si, ti = line
            padding = [0 for _ in range(max_len - len(chars))]
            chars.append(ch.extend(padding))
            char_idx.append(ci.extend(padding))
            seg_idx.append(si.extend(padding))
            tag_idx.append(ti.extend(padding))
        return chars, char_idx, seg_idx, tag_idx

    def iter_batch(self, shuffle=False):
        if shuffle:  # 在sort时打乱过一次, 考虑对长度整理的要求，可以在单个batch内进行shuffle
            random.shuffle(self.batch_data)
        for i in range(self.num_batches):
            yield self.batch_data[i]