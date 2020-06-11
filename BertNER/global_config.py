#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   global_config.py
@Author  :   Racle
@Version :   1.0
@Desc    :   None
'''

from multiprocessing import cpu_count

global_configs = {
  # TODO：文件IO位置设置
  "best_model_dir": "outputs/best_model",
  "output_dir": "outputs/",
  "overwrite_output_dir": False,
  "tensorboard_dir": None,
  "cache_dir": "cache_dir/",

  "config": {},         # 传入BertConfig的参数
  "do_lower_case": False,
  "encoding": None,

  # TODO：train 参数设置
  "train_batch_size": 4,
  "num_train_epochs": 1,

  "learning_rate": 4e-5,
  "adam_epsilon": 1e-6,  # 在分母上增加一项，以提高数值稳定性
  "weight_decay": 0,

  # 两种方式任选其一
  "warmup_ratio": 0,
  "warmup_steps": 1000,

  "max_grad_norm": 1.0,
  "max_seq_length": 512,

  # 训练策略
  "n_gpu": 1,
  "fp16": True,
  "fp16_opt_level": "O1",  # Mixed Precision (recommended for typical use)  https://nvidia.github.io/apex/amp.html，
  "gradient_accumulation_steps": 3,

  "use_early_stopping": True,
  "early_stopping_consider_epochs": False, # 按epoch进行early_stopping
  "early_stopping_delta": 0,               # early_stopping的下降阈值
  "early_stopping_metric": "eval_loss",    # early_stopping的参照指标
  "early_stopping_metric_minimize": True,  # 最小化early_stopping的参照指标，还是最大化early_stopping的参照指标
  "early_stopping_patience": 10,

  "manual_seed": None,
  "save_best_model": True,
  "save_eval_checkpoints": True,
  "save_model_every_epoch": True,
  "save_steps": 2000,

  "silent": False,
  "logging_steps": 50,
  "no_cache": False,  # 不保存使用处理好的特征缓存
  "no_save": False,   # 训练时不保存模型

  "use_cached_eval_features": False,  # 使用缓存的处理好的验证数据，mode=="eval"及no_cache==False时有效

  "use_multiprocessing": True,
  "process_count": cpu_count()-2 if cpu_count() > 2 else 1,
  "multiprocessing_chunksize": 500,
  "reprocess_input_data": True,  # 不使用处理好的特征，从输入数据再次处理特征

  # TODO：evaluate 参数设置
  # Evaluate 确保在单GPU环境下运行
  "eval_batch_size": 4,
  "evaluate_during_training": False,         # 训练过程中evaluate
  "evaluate_during_training_steps": 2000,    # 每2000步，evaluate。使用early_stopping时，同时进行early_stopping检测
  "evaluate_during_training_verbose": False,

  # TODO：Weight & Bias包参数设置
  # 注意使用该工具包前，先在command line进行登录：wandb login [tokens]。tokens来自 https://app.wandb.ai/home
  "wandb_project": None,
  "wandb_kwargs": {}
}
