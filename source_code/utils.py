#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Author  :   Racle
@Version :   1.0
@Desc    :   None
'''

import os
import shutil
from conlleval import return_report

import jieba
jieba.initialize()
jieba_dict_prepare()


def clean(flags):
    """
    Clean current folder
    remove saved model and training log
    """
    if os.path.isfile(flags.vocab_file):
        os.remove(flags.vocab_file)

    if os.path.isfile(flags.map_file):
        os.remove(flags.map_file)

    if os.path.isdir(flags.ckpt_path):
        shutil.rmtree(flags.ckpt_path)

    if os.path.isdir(flags.summary_path):
        shutil.rmtree(flags.summary_path)

    if os.path.isdir(flags.result_path):
        shutil.rmtree(flags.result_path)

    if os.path.isdir("log"):
        shutil.rmtree("log")

    if os.path.isdir("__pycache__"):
        shutil.rmtree("__pycache__")

    if os.path.isfile(flags.config_file):
        os.remove(flags.config_file)

    if os.path.isfile(flags.vocab_file):
        os.remove(flags.vocab_file)


def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    IOB2 -- 即为修正的IOB格式
    """
    for i, tag in enumerate(tags):
        if tag == 'O':  # “O” 标记
            continue
        split = tag.split('-')
        # 非法格式
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        # “B-xxx” 的标记
        if split[0] == 'B':
            continue
        # 此时为 “I-xxx” 的标记，修正没有“B-xxx”标记的“I-xxx”
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        # 此时为 “I-xxx” 的标记，正确的“I-xxx”的情况继续
        elif tags[i - 1][1:] == tag[1:]:
            continue
        # 当个 “I-xxx” 标记的情况
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def iob_iobes(tags):
    "I - 中间字； B - 起始字； E - 结束字； S - 单字； O - 非entity"
    new_tags = []
    for i, tag in enumerate(tags):
        if tag=='O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if (i+1) != len(tags) and tags[i+1].strip('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-')
        elif tag.split('-')[0] == 'I':
            if (i+1) != len(tags) and tags[i+1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('# >>> Invalid format !!! <<< #')
    return new_tags


def iobes_iob(tags):
    new_tags = []
    for i, tag in enumerate(tags):
        tag_prefix = tag.split('-')[0]
        if tag_prefix == 'B':
            new_tags.append(tag)
        elif tag_prefix == 'I':
            new_tags.append(tag)
        elif tag_prefix == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag_prefix == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag_prefix == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags


def load_word2vec(emb_path, id_to_word, word_dim, weighs):
    "加载预训练的词向量，注意维度匹配"
    # 1. read pretrained weights
    print('=> Loading pretrained embeddings from {}...'.format(emb_path))
    pre_trained = {}
    emb_invalid = 0
    for i, line in enumerate(codecs.open(emb_path, 'r', 'utf-8')):
        line = line.rstrip().split()
        if len(line) == word_dim + 1:
            pre_trained[line[0]] = np.array(
                [float(x) for x in line[1:]]).astype(np.float32)
        else:
            emb_invalid += 1
    if emb_invalid > 0:
        print('=》 WARNING: %i invalid lines' % emb_invalid)

    # 2. weights assignment
    c_found = 0
    n_words = len(id_to_word)
    for i in range(n_words):
        word = id_to_word[i]
        if word in pre_trained:
            weights[i] = pre_trained[word]
            c_found += 1
        elif word.lower() in pre_trained:
            weights[i] = pre_trained[word.lower()]
            c_found += 1
        elif re.sub('\d', '0', word.lower()) in pre_trained:
            weights[i] = pre_trained[re.sub('\d', '0', word.lower())]
            c_found += 1
    print('=> Loaded %i pretrained embeddings.' % len(pre_trained))
    print('=> %i / %i words have been initialized with pretrained embeddings.' % (c_found, n_words))
    return weights


def test_ner(results, path):
    """
    Run perl script to evaluate model
    """
    output_file = os.path.join(path, "ner_predict.utf8")
    with open(output_file, "w", encoding='utf8') as f:
        to_write = []
        for res in results:
            for line in res:
                to_write.append(line + "\n")
            to_write.append("\n")
        f.writelines(to_write)
    eval_lines = return_report(output_file)
    return eval_lines


def input_from_line(line, char_to_id):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    line = full_to_half(line)
    line = replace_html(line)
    inputs = list()
    inputs.append([line])
    line.replace(" ", "$")
    inputs.append([[char_to_id[char] if char in char_to_id else char_to_id["<UNK>"]
                   for char in line]])
    inputs.append([get_seg_features(line)])
    inputs.append([[]])
    return inputs


def full_to_half(s):
    """
    Convert full-width character to half-width one
    """
    n = []
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        char = chr(num)
        n.append(char)
    return ''.join(n)


def replace_html(s):
    s = s.replace('&quot;','"')
    s = s.replace('&amp;','&')
    s = s.replace('&lt;','<')
    s = s.replace('&gt;','>')
    s = s.replace('&nbsp;',' ')
    s = s.replace("&ldquo;", "")
    s = s.replace("&rdquo;", "")
    s = s.replace("&mdash;","")
    s = s.replace("\xa0", " ")
    return(s)


def get_seg_features(string):
    """
    Segment text with jieba
    features are represented in bies format
    s donates single word
    注意：若jieba中的词表没有永久添加目标NER词汇，需要在使用前进行手动添加。
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


def jieba_dict_prepare(dict_path="./source_data/DICT_NOW.csv"):
    "根据语料资源，向jieba中添加自定义的词。在使用jieba进行分词前可使用。"
    import jieba, csv

    # 数据说明：
    #     DICT_NOW.csv:
    #           所有标记对应的语言组成的dictionary
    #           这一部分只需要将***.txt中的entity mention和entity category进行对应输出即可
    #           得到DICT_NOW.csv文件。（实际中可以在医药网站或者医学百科中爬取一些医学类entity）
    dics = csv.reader(open(dict_path, 'r', encoding='utf8'))

    # 利用jieba自定义分词，进行专有名词输入
    # 将识别对象加入jieba识别词表，标记视为词性
    for row in dics:
        if len(row) == 2:
            jieba.add_word(row[0].strip(), tag=row[1].strip())
            # 强制加入词为一个joined整体
            jieba.suggest_freq(row[0].strip())


def result_to_json(string, tags):
    item = {"string": string, "entities": []}
    entity_name = ""
    entity_start = 0
    idx = 0
    for char, tag in zip(string, tags):
        prefix = tag[0]
        if prefix == "S":
            item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":tag[2:]})
        elif prefix == "B":
            entity_name += char
            entity_start = idx
        elif prefix == "I":
            entity_name += char
        elif prefix == "E":
            entity_name += char
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2:]})
            entity_name = ""
        else:
            entity_name = ""
            entity_start = idx
        idx += 1
    return item