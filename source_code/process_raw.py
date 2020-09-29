#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   process_raw.py
@Author  :   Racle
@Version :   1.0
@Desc    :   None
'''

import os, jieba, csv
import time
import jieba.posseg as pseg

# os.sep：根据系统自适应path分隔符
c_root = os.getcwd() + os.sep + "source_data" + os.sep
data_root = os.getcwd() + os.sep + "data" + os.sep

# 处理后数据存放文件
dev = open(os.path.join(data_root, "example.dev"), 'w', encoding='utf8')
train = open(os.path.join(data_root, "example.train"), 'w', encoding='utf8')
test = open(os.path.join(data_root, "example.test"), 'w', encoding='utf8')

# tagging标记集合：in 操作 list--O(n); set--O(1)
biaoji = set([
    'DIS', 'SYM', 'SGN', 'TES', 'DRU', 'SUR', 'PRE', 'PT', 'Dur', 'TP', 'REG',
    'ORG', 'AT', 'PSB', 'DEG', 'FW', 'CL'
])

# 标点符号集合：作为一个序列的结尾
fuhao = set(['。', '?', '？', '!', '！'])

# 数据说明：
#     ***orginal.txt
#   		原始纯文本文件（也就是NER的输入文件）
#     ***.txt
#   		对应同名原始文件的标注结果，分四个域
#   			entity mention: 识别出的实体名称
#   			pos_b: entity mention在文件中的起始位置（从0开始编号）
#   			pos_e: entity mention在文件中的终止位置
#   			category：entity所属的类别
#     DICT_NOW.csv:
#           所有标记对应的语言组成的dictionary
#           这一部分只需要将***.txt中的entity mention和entity category进行对应输出即可
#           得到DICT_NOW.csv文件。（实际中可以在医药网站或者医学百科中爬取一些医学类entity）
dics = csv.reader(open("./source_data/DICT_NOW.csv", 'r', encoding='utf8'))


# =======================================================================================
# 利用jieba自定义分词，进行专有名词输入
# 将识别对象加入jieba识别词表，标记视为词性
for row in dics:
    if len(row) == 2:
        jieba.add_word(row[0].strip(), tag=row[1].strip())
        # 强制加入词为一个joined整体
        jieba.suggest_freq(row[0].strip())

# =======================================================================================
# 读取目标文件，进行IOB格式的标记，并写入dev、train、test文件
split_num = 0
start_time = time.time()

for file in os.listdir(c_root):
    if "txtoriginal.txt" in file:
        fp = open(c_root + file, 'r', encoding='utf8')
        for line in fp:
            split_num += 1
            words = pseg.cut(line)    # 带词性切词
            # key: word； value： part of speech
            for key, value in words:
                if value.strip() and key.strip():
                    # index值用于划分dev、train、test
                    index = str(1) if split_num % 15 < 2 else str(2) \
                        if split_num % 15 > 1 and split_num % 15 < 4 else str(3)
                    if value.strip() not in biaoji:
                        value = 'O'
                        # 按字标记
                        for achar in key.strip():
                            if achar and achar.strip() in fuhao:
                                string = achar + " " + value.strip() + "\n" + "\n"
                                # 划分dev、train、test
                                dev.write(string) if index == '1' else \
                                test.write(string) if index == '2' else \
                                train.write(string)
                            elif achar.strip() and achar.strip() not in fuhao:
                                string = achar + " " + value.strip() + "\n"
                                # 划分dev、train、test
                                dev.write(string) if index == '1' else \
                                test.write(string) if index == '2' else \
                                train.write(string)

                    elif value.strip() in biaoji:
                        begin = 0
                        for char in key.strip():
                            # 开始字，以 B- 开头
                            if begin == 0:
                                begin += 1
                                string1 = char + ' ' + 'B-' + value.strip() + '\n'
                                if index == '1':
                                    dev.write(string1)
                                elif index == '2':
                                    test.write(string1)
                                elif index == '3':
                                    train.write(string1)
                                else:
                                    pass
                            else:   # 开始字之后，以 I- 开头
                                string1 = char + ' ' + 'I-' + value.strip() + '\n'
                                if index == '1':
                                    dev.write(string1)
                                elif index == '2':
                                    test.write(string1)
                                elif index == '3':
                                    train.write(string1)
                                else:
                                    pass
                    else:
                        continue

# 处理后文件储存格式为：
# 患 O
# 者 O
# 以 O
# 腰 O
# 痛 O
# 伴 O
# 双 B-REG
# 下 I-REG
# 肢 I-REG
# 疼 B-SYM
# 痛 I-SYM
# 半 O
# 年 O

end_time = time.time()
print(("IOB tagging used time is {} s. \n".format(end_time - start_time)))