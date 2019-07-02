# NER_project
Electronic medical record NER project using idCNN + CRF.

该项目主要使用idCNN + CRF loss的结构解决电子病历中的NER问题，在测试数据集上的F1 score在94%左右。

---
## idCNN
使用bi-LSTM的问题在于训练速度较慢，并且对于本项目的局部实体识别问题，idCNN对局部实体的卷积编码比bi-LSTM的长距离依赖关系编码
相对而言达到更好的实体识别效果，同时提升了训练推理的效率。[idCNN详解](膨胀卷积神经网络dilated convolutions.ipynb)

---
## CRF
模型输入的标记格式为IOBES（[详见文件](电子病历实体识别项目.ipynb)），通过jieba工具以词性的方式进行标注。CRF计算文本每个字的
标记概率，通过gold label和pred label的log likelyhood loss反向传播求解最优的transfer matrix。这比简单的计算cross entropy
 loss多了一个全局的转移weight，通过viterbi decode可以得到更优的结果。文件夹下有CRF相关的简单介绍。
 > ![pic1](source_code/pic/pic1.png)
 > ![pic2](source_code/pic/pic2.png)
 
 ---
 ## 项目结构
 > source_code:
      - gen_data.py: 生成训练、开发、测试文件
      - main_local.py: 本地训练文件
      - main_flask.py: 使用flask部署
      - model.py: 模型文件
 > package:
      - main.py: 训练好的lib package使用入口
      
