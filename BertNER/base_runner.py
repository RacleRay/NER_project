#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   base_runer.py
'''

import json
import logging
import math
import os
import random
import pickle

import numpy as np
import pandas as pd
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from tqdm.auto import tqdm, trange
import wandb

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tensorboardX import SummaryWriter

from transformers import (
    AutoModelForTokenClassification,
    AlbertForTokenClassification,
    BertForTokenClassification,
    DistilBertForTokenClassification,
    ElectraForTokenClassification,
    RobertaForTokenClassification,
    XLNetForTokenClassification
)

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoTokenizer,
    AlbertConfig,
    AlbertTokenizer,
    BertConfig,
    BertTokenizer,
    DistilBertConfig,
    DistilBertTokenizer,
    ElectraConfig,
    ElectraTokenizer,
    RobertaConfig,
    RobertaTokenizer,
    XLNetConfig,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)
from custom_models.bert_lstm_crf import BertLstmCrf
from custom_models.albert_lstm_crf import AlbertLstmCrf
from custom_models.electra_lstm_crf import ElectraLstmCrf
from custom_model_config import bert_lstm_crf_config

from base_utils import (
    InputExample,
    convert_examples_to_features,
    get_examples_from_df,
    get_labels,
    read_examples_from_file,
)
from global_config import global_configs


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BertLstmCrfConfig():
    hidden_dropout_prob = 0.2
    num_layers = 1
    lstm_dropout = 0.2

    label_encoding = "BIO"
    freez_prrtrained = False


class BaseRunner:
    def __init__(
            self,
            model_type,
            model_file,
            label_format_path=None,
            args=None,
            use_cuda=True,
            cuda_device=-1,
            **kwargs
    ):
        """
        Initializes a BaseRunner model.

        Args:
            model_type: MODEL_CLASSES中的模型名称，bert，xlnet，roberta，distilbert，albert，electra, bert_lstm_crf, albert_lstm_crf
            model_file: 预训练的模型文件路径，或者已经fine-tune过的模型保存路径，或者在huggingface注册的模型名称
                        https://huggingface.co/models?search=chinese
            label_format_path: NER标记格式的文件路径（txt或其他open可读的）.
            weight (optional): 不平衡类别时，可选的类别权重list
            args (optional): 模型参数设置dict, 会在默认的global_config基础上修改。
            use_cuda (optional): 是否用GPU
            cuda_device (optional): 默认使用第一个识别到的设备
            **kwargs (optional): 传入transformers的from_pretrained方法的参数
        """
        MODEL_CLASSES = {
            "auto": (AutoConfig, AutoTokenizer, AutoModelForTokenClassification),
            "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
            "bert_lstm_crf": (BertConfig, BertLstmCrf, BertTokenizer),
            "xlnet": (XLNetConfig, XLNetForTokenClassification, XLNetTokenizer),
            "roberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer),
            "distilbert": (DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizer),
            # "albert": (AlbertConfig, AlbertForTokenClassification, AlbertTokenizer),
            # 由于 albert_chinese_* 模型没有用 sentencepiece.
            # 用AlbertTokenizer加载不了词表，因此需要改用BertTokenizer
            # https://huggingface.co/voidful/albert_chinese_xxlarge
            "albert": (AlbertConfig, AlbertForTokenClassification, BertTokenizer),
            "albert_lstm_crf": (AlbertConfig, AlbertLstmCrf, BertTokenizer),
            "electra": (ElectraConfig, ElectraForTokenClassification, ElectraTokenizer),
            "electra_lstm_crf": (ElectraConfig, ElectraLstmCrf, ElectraTokenizer),
        }

        if args and "manual_seed" in args:
            random.seed(args["manual_seed"])
            np.random.seed(args["manual_seed"])
            torch.manual_seed(args["manual_seed"])
            if "n_gpu" in args and args["n_gpu"] > 0:
                torch.cuda.manual_seed_all(args["manual_seed"])

        self.args = {}
        self.args = {"classification_report": False} # seqeval包的设置
        self.args.update(global_configs)

        self.args.update(global_configs)
        saved_model_args = self._load_model_args(model_file)
        if saved_model_args:
            self.args.update(saved_model_args)
        if args:
            self.args.update(args)

        self.args["model_file"] = model_file
        self.args["model_type"] = model_type

        self.labels = get_labels(label_format_path)
        self.num_labels = len(self.labels)
        self.idx2tag = {i: tag for i, tag in enumerate(self.labels)}

        # 使用自定义模型时的参数
        extra_config = bert_lstm_crf_config
        extra_config["idx2tag"] = self.idx2tag

        self.results = {}

        # ### TODO:导入预训练模型、config ###
        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        if self.num_labels:
            self.config = config_class.from_pretrained(model_file, num_labels=self.num_labels, **self.args["config"])
            self.num_labels = self.num_labels
        else:
            self.config = config_class.from_pretrained(model_file, **self.args["config"])
            self.num_labels = self.config.num_labels
        self.config.output_hidden_states = False
        self.config.output_attentions = False

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError("没有可用的CUDA环境.")
        else:
            self.device = "cpu"

        # TODO: 初始化预训练模型的tokenizer
        self.tokenizer = tokenizer_class.from_pretrained(
            model_file, do_lower_case=self.args["do_lower_case"], **kwargs
        )
        # 计算crf loss时不要的token ids
        self.ignore_ids = self._get_ignore_ids()

        # TODO: 导入预训练模型
        if "lstm_crf" in model_type:
            self.model = model_class.from_pretrained(model_file,
                                                     config=self.config,
                                                     extra_config=bert_lstm_crf_config,
                                                     ignore_ids=self.ignore_ids,
                                                     **kwargs)
        else:
            self.model = model_class.from_pretrained(model_file, config=self.config, **kwargs)

        if not use_cuda:
            self.args["fp16"] = False

        # pad token of label seq
        self.pad_token_label_id = CrossEntropyLoss().ignore_index  # -100

    def train_model(
            self,
            train_data,
            output_dir=None,
            show_running_loss=True,
            args=None,
            eval_data=None,
            verbose=True,
            **kwargs,
    ):
        """
        Args:
            train_data: Pandas Dataframe. 需要有三列
                                            “words”-一个词
                                            “labels”-词对应的tag
                                            “sentence_id”-词所在句子的id，用来groupby
                        txt. 每一行为 “word tag”, 每句话之间用space空行隔开。
            output_dir: 模型保存路径，在self.args['output_dir']，可以修改. self.args读取config文件的设置.
            show_running_loss (optional): Defaults True.
            args (optional): 对config中的设置进行修改.
            eval_data (optional): ‘evaluate_during_training‘为True时，传入进行评估运行效果. 传入要求与train_data一致
            **kwargs: metric 函数对象. 比如传入 f1=sklearn.metrics.f1_score。函数要求接受pretict和true两个值。
        Returns:
            None
        """
        # 检查参数设置
        if args: self.args.update(args)
        if not output_dir: output_dir = self.args["output_dir"]
        if self.args["silent"]: show_running_loss = False
        if self.args["evaluate_during_training"] and eval_data is None:
            raise ValueError("请指定eval_data")
        if os.path.exists(output_dir) and os.listdir(output_dir) and not self.args["overwrite_output_dir"]:
            raise ValueError("output_dir ({}) 已存在. 设置 --overwrite_output_dir 覆盖文件.".format(output_dir))

        # 输入数据读取
        train_dataset = self.load_and_cache_examples(train_data, verbose=verbose)

        os.makedirs(output_dir, exist_ok=True)

        self.model.to(self.device)
        global_step, total_running_loss = self.train(
            train_dataset,
            output_dir,
            show_running_loss=show_running_loss,
            eval_data=eval_data,
            verbose=verbose,
            **kwargs,
        )
        self._save_model(model=self.model)  # 可以传入优化器，保存当前优化器状态
        if verbose:
            logger.info(" 训练 {} 结束. Saved to {}.".format(self.args["model_type"], output_dir))
            logger.info(" 当前global step： {}. 运行平局loss： {}.".format(global_step, total_running_loss))

    def train(
            self,
            train_dataset,
            output_dir,
            show_running_loss=True,
            eval_data=None,
            verbose=True,
            **kwargs,
    ):
        # ==============================
        # TODO: settings
        # ==============================
        device = self.device
        model = self.model
        args = self.args

        global_step = 0
        total_running_loss, previous_log_running_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(int(args["num_train_epochs"]), desc="Epoch", disable=args["silent"], mininterval=0)
        epoch_number = 0
        best_eval_metric = None
        early_stopping_counter = 0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0

        tb_writer = SummaryWriter(logdir=args["tensorboard_dir"])
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args["train_batch_size"])

        # num_training_steps
        # gradient_accumulation 累积一定数量loss.backward，再更新模型参数optimizer.step()
        t_total = len(train_dataloader) // args["gradient_accumulation_steps"] * args["num_train_epochs"]

        no_decay = ["bias", "LayerNorm.weight"]
        if 'lstm_crf' not in self.model_type:
            optimizer_grouped_parameters = [
                {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                 "weight_decay": args["weight_decay"]},
                {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0},
            ]
        else:
            # =================== TODO：不同层设置不同的学习率 ===================
            pretrained_model_params = model.pretrained.named_parameters()
            lstm_param_optimizer = model.bilstm.named_parameters()
            crf_param_optimizer = model.crf.named_parameters()
            linear_param_optimizer = model.classifier.named_parameters()
            optimizer_grouped_parameters = [
                {'params': [p for n, p in pretrained_model_params if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01,
                 'lr': args["learning_rate"]},
                {'params': [p for n, p in pretrained_model_params if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0,
                 'lr': args["learning_rate"]},
                {'params': [p for n, p in lstm_param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01,
                 'lr': 0.001},
                {'params': [p for n, p in lstm_param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0,
                 'lr': 0.001},
                {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01,
                 'lr': 0.001},
                {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0,
                 'lr': 0.001},
                {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01,
                 'lr': 0.001},
                {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0,
                 'lr': 0.001}
            ]
            # =================================================================

        # warm up
        warmup_steps = math.ceil(t_total * args["warmup_ratio"])
        args["warmup_steps"] = warmup_steps if args["warmup_steps"] == 0 else args["warmup_steps"]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args["learning_rate"], eps=args["adam_epsilon"])
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args["warmup_steps"], num_training_steps=t_total)

        # 半精度
        if args["fp16"]:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("需要安装 apex ：https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args["fp16_opt_level"])

        # 模型复制，数据并行，梯度累计
        if args["n_gpu"] > 1:
            model = torch.nn.DataParallel(model)

        # model file
        if args["model_file"] and os.path.exists(args["model_file"]):
            try:  # 读取已经fine-tune过的模型文件
                # checkpoint中读取global_step
                checkpoint_suffix = args["model_file"].split("/")[-1].split("-")
                if len(checkpoint_suffix) > 2:
                    checkpoint_suffix = checkpoint_suffix[1]
                else:
                    checkpoint_suffix = checkpoint_suffix[-1]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (len(train_dataloader) // args["gradient_accumulation_steps"])
                steps_trained_in_current_epoch = global_step % (
                        len(train_dataloader) // args["gradient_accumulation_steps"])

                logger.info(">>> 加载checkpoint，更新global_step.")
                logger.info(">>> 已训练 %d 轮.", epochs_trained)
                logger.info(">>> 当前global_step： %d.", global_step)
                logger.info(">>> 跳过当前epoch %d steps in.", steps_trained_in_current_epoch)
            except ValueError:  # 读取预训练文件，fine-tuning
                logger.info(">>> 开始 fine-tuning.")

        # 设置训练过程中，评价模型的指标
        if args["evaluate_during_training"]:
            training_progress_scores = self._create_training_progress_scores(**kwargs)

        # weight & bias：记录模型参数变化
        if args["wandb_project"]:
            wandb.init(project=args["wandb_project"], config={**args}, **args["wandb_kwargs"])
            wandb.watch(self.model)

        # ================================
        # TODO: 训练框架流程
        # ================================
        model.train()
        for _ in train_iterator:
            if epochs_trained > 0:  # 处理加载checkpoint的情况
                epochs_trained -= 1
                continue
            # 迭代一个epoch
            for step, batch in enumerate(tqdm(train_dataloader, desc="Current iteration", disable=args["silent"])):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                batch = tuple(t.to(device) for t in batch)

                # 运行模型前向计算
                inputs = self._get_inputs_dict(batch)  # 不同模型输入有一些差异
                # ================================
                # TODO: 修改模型代码
                # ================================
                outputs = model(**inputs)

                # ### BP训练设计 ###
                # outputs： ((loss), logits, (hidden_states), (attentions))
                loss = outputs[0]
                if args["n_gpu"] > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                current_loss = loss.item()
                if show_running_loss and step % 100 == 0:
                    print("\rRunning loss: %f" % loss, end="")

                # Normalize our loss (if averaged)
                if args["gradient_accumulation_steps"] > 1:
                    loss = loss / args["gradient_accumulation_steps"]

                if args["fp16"]:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                total_running_loss += loss.item()
                if (step + 1) % args["gradient_accumulation_steps"] == 0:
                    if args["fp16"]:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args["max_grad_norm"])
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args["max_grad_norm"])

                    optimizer.step()  # gradient_accumulation_steps 更新一次参数
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    # 记录参数变化
                    if args["logging_steps"] > 0 and global_step % args["logging_steps"] == 0:
                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar("loss", (total_running_loss - previous_log_running_loss) / args["logging_steps"], global_step)
                        previous_log_running_loss = total_running_loss
                        if args["wandb_project"]:
                            wandb.log({"Training loss": current_loss, "lr": scheduler.get_lr()[0], "global_step": global_step})

                    # Save model checkpoint
                    if args["save_steps"] > 0 and global_step % args["save_steps"] == 0:
                        output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))
                        self._save_model(output_dir_current, optimizer, scheduler, model=model)

                    # ### Evaluate every xxx steps ###
                    if args["evaluate_during_training"] and (args["evaluate_during_training_steps"] > 0
                                                             and global_step % args["evaluate_during_training_steps"] == 0):
                        # Evaluate 确保在单GPU环境下运行
                        results, model_outputs, wrong_preds = self.eval_model(
                            eval_data,
                            verbose=verbose and args["evaluate_during_training_verbose"],
                            silent=True,
                            **kwargs,
                        )
                        for key, value in results.items():  # results: compute_metrics输出的字典
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)  # tensorboard记录

                        # 保存Evaluate时的model状态
                        output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))
                        if args["save_eval_checkpoints"]:
                            self._save_model(output_dir_current, optimizer, scheduler, model=model, results=results)

                        # 保存log
                        training_progress_scores["global_step"].append(global_step)
                        training_progress_scores["train_loss"].append(current_loss)
                        for key in results:
                            training_progress_scores[key].append(results[key])
                        report = pd.DataFrame(training_progress_scores)
                        report.to_csv(os.path.join(output_dir, "training_progress_scores.csv"), index=False)

                        # weight & bias
                        if args["wandb_project"]:
                            wandb.log({metric: values[-1] for metric, values in training_progress_scores.items()})

                        # ### Early stop every xxx steps ###
                        if not best_eval_metric:
                            best_eval_metric = results[args["early_stopping_metric"]]  # early_stopping_metric: early_stopping的参照指标
                            self._save_model(args["best_model_dir"], optimizer, scheduler, model=model, results=results)
                        # 最小化early_stopping的参照指标为目标
                        if best_eval_metric and args["early_stopping_metric_minimize"]:
                            if (
                                    results[args["early_stopping_metric"]] - best_eval_metric < args["early_stopping_delta"]
                            ):
                                best_eval_metric = results[args["early_stopping_metric"]]
                                self._save_model(args["best_model_dir"], optimizer, scheduler, model=model, results=results)
                                early_stopping_counter = 0
                            else:
                                if args["use_early_stopping"]:
                                    if early_stopping_counter < args["early_stopping_patience"]:
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(f" Early stop指标 {args['early_stopping_metric']} 没有提升.")
                                            logger.info(f" 当前已执行early stop count: {early_stopping_counter}")
                                            logger.info(f" Early stop最大容忍轮次: {args['early_stopping_patience']}")
                                    else:
                                        if verbose:
                                            logger.info(f" 达到Early stop最大容忍轮次.")
                                            logger.info(" 停止训练@.")
                                            train_iterator.close()
                                        return global_step, total_running_loss / global_step
                        else:  # 最大化early_stopping的参照指标为目标
                            if (results[args["early_stopping_metric"]] - best_eval_metric > args["early_stopping_delta"]):
                                best_eval_metric = results[args["early_stopping_metric"]]
                                self._save_model(args["best_model_dir"], optimizer, scheduler, model=model, results=results)
                                early_stopping_counter = 0
                            else:
                                if args["use_early_stopping"]:
                                    if early_stopping_counter < args["early_stopping_patience"]:
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(f" Early stop指标 {args['early_stopping_metric']} 没有提升.")
                                            logger.info(f" 当前已执行early stop count: {early_stopping_counter}")
                                            logger.info(f" Early stop最大容忍轮次: {args['early_stopping_patience']}")
                                    else:
                                        if verbose:
                                            logger.info(f" 达到Early stop最大容忍轮次.")
                                            logger.info(" 停止训练@.")
                                            train_iterator.close()
                                        return global_step, total_running_loss / global_step
            # 完成一个epoch
            epoch_number += 1
            output_dir_current = os.path.join(output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number))

            # save model
            if args["save_model_every_epoch"] or args["evaluate_during_training"]:
                os.makedirs(output_dir_current, exist_ok=True)
            if args["save_model_every_epoch"]:
                self._save_model(output_dir_current, optimizer, scheduler, model=model)

            # ### Evaluate every epoch ###
            # 逻辑和 Evaluate every xxx steps 一样
            if args["evaluate_during_training"]:
                results, _, _ = self.eval_model(
                    eval_data, verbose=verbose and args["evaluate_during_training_verbose"], silent=True, **kwargs
                )
                self._save_model(output_dir_current, optimizer, scheduler, results=results)

                training_progress_scores["global_step"].append(global_step)
                training_progress_scores["train_loss"].append(current_loss)
                for key in results:
                    training_progress_scores[key].append(results[key])
                report = pd.DataFrame(training_progress_scores)
                report.to_csv(os.path.join(output_dir, "training_progress_scores.csv"), index=False)

                if args["wandb_project"]:
                    wandb.log(self._get_last_metrics(training_progress_scores))

                # ### Early stop every epoch ###
                if not best_eval_metric:
                    best_eval_metric = results[args["early_stopping_metric"]]
                    self._save_model(args["best_model_dir"], optimizer, scheduler, model=model, results=results)
                # 最小化early_stopping的参照指标为目标
                if best_eval_metric and args["early_stopping_metric_minimize"]:
                    if results[args["early_stopping_metric"]] - best_eval_metric < args["early_stopping_delta"]:
                        best_eval_metric = results[args["early_stopping_metric"]]
                        self._save_model(args["best_model_dir"], optimizer, scheduler, model=model, results=results)
                        early_stopping_counter = 0
                    else:
                        if args["use_early_stopping"] and args["early_stopping_consider_epochs"]:  # 在epoch维度上进行early_stopping
                            if early_stopping_counter < args["early_stopping_patience"]:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(f" Early stop指标 {args['early_stopping_metric']} 没有提升.")
                                    logger.info(f" 当前已执行early stop count: {early_stopping_counter}")
                                    logger.info(f" Early stop最大容忍轮次: {args['early_stopping_patience']}")
                            else:
                                if verbose:
                                    logger.info(f" 达到Early stop最大容忍轮次.")
                                    logger.info(" 停止训练@.")
                                    train_iterator.close()
                                return global_step, total_running_loss / global_step
                else:  # 最大化early_stopping的参照指标为目标
                    if results[args["early_stopping_metric"]] - best_eval_metric > args["early_stopping_delta"]:
                        best_eval_metric = results[args["early_stopping_metric"]]
                        self._save_model(args["best_model_dir"], optimizer, scheduler, model=model, results=results)
                        early_stopping_counter = 0
                    else:
                        if args["use_early_stopping"] and args["early_stopping_consider_epochs"]:
                            if early_stopping_counter < args["early_stopping_patience"]:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(f" Early stop指标 {args['early_stopping_metric']} 没有提升.")
                                    logger.info(f" 当前已执行early stop count: {early_stopping_counter}")
                                    logger.info(f" Early stop最大容忍轮次: {args['early_stopping_patience']}")
                            else:
                                if verbose:
                                    logger.info(f" 达到Early stop最大容忍轮次.")
                                    logger.info(" 停止训练@.")
                                    train_iterator.close()
                                return global_step, total_running_loss / global_step

        return global_step, total_running_loss / global_step

    def eval_model(self, eval_data, output_dir=None, verbose=True, silent=False, **kwargs):
        """
        评估模型，确保在单GPU环境下运行。
        Args:
            eval_data: 两种格式：
                       Pandas Dataframe. 需要有三列
                                            “words”-一个词
                                            “labels”-词对应的tag
                                            “sentence_id”-词所在句子的id，用来groupby
                       txt. 每一行为 “word tag”, 每句话之间用space空行隔开。
            output_dir: 保存结果的路径
            verbose: 控制台显示
            silent: 显示进度条.
            **kwargs: metric 函数对象. 比如传入 f1=sklearn.metrics.f1_score。函数要求接受pretict和true两个值。
        Returns:
            result: Dictionary -- evaluation results.
            model_outputs: List -- 输入每一个样例的预测结果.
            wrong_preds: List -- 错误预测样例.
        """
        if not output_dir:
            output_dir = self.args["output_dir"]

        eval_dataset = self.load_and_cache_examples(eval_data, evaluate=True)

        self.model.to(self.device)
        result, preds, model_outputs, preds_list = self.evaluate(
            eval_dataset, output_dir, verbose=verbose, silent=silent, **kwargs)
        self.results.update(result)

        if verbose:
            logger.info(self.results)

        return result, model_outputs, preds_list

    def evaluate(self, eval_dataset, output_dir, verbose=True, silent=False, **kwargs):
        """
        **kwargs: metric 函数对象. 比如传入 f1=sklearn.metrics.f1_score。函数要求接受pretict和true两个值。
        """
        device = self.device
        model = self.model
        args = self.args
        pad_token_label_id = self.pad_token_label_id
        eval_output_dir = output_dir

        results = {}

        # ### 输入数据读取 ###

        # ### data loader ###
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args["eval_batch_size"])

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        os.makedirs(eval_output_dir, exist_ok=True)

        # ### Evaluate ###
        model.eval()
        for batch in tqdm(eval_dataloader, disable=args["silent"] or silent):
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = self._get_inputs_dict(batch)
                outputs = model(**inputs)  # outputs： ((loss), logits, (hidden_states), (attentions))
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)  # 不是list的append
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        model_outputs = preds
        preds = np.argmax(preds, axis=1)

        label_map = {i: label for i, label in enumerate(self.labels)}

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        extra_metrics = {}
        for metric, func in kwargs.items():
            extra_metrics[metric] = func(out_label_list, preds_list)
        # 结果导出
        result = self.compute_metrics(preds_list, out_label_ids, **kwargs)
        result["eval_loss"] = eval_loss
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w", encoding="utf-8") as writer:
            if args["classification_report"]:
                cls_report = classification_report(out_label_list, preds_list)
                writer.write("{}\n".format(cls_report))
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))

        return results, model_outputs, preds_list

    def load_and_cache_examples(self, data, evaluate=False, no_cache=False, to_predict=None, verbose=True):
        """
        将InputExample转化为Dataset，可保存和读取处理后的InputFeatures. train() 和 eval() 的辅助方法.

        Returns:
            dataset -- InputExample的Dataset， (input_ids, input_mask, segment_ids, label_ids).
        """
        args = self.args
        tokenizer = self.tokenizer

        process_count = args["process_count"]  # 调用的CPU数量
        if not no_cache:
            no_cache = args["no_cache"]  # 是否启用特征缓存

        mode = "eval" if evaluate else "train"
        os.makedirs(self.args["cache_dir"], exist_ok=True)
        # 设置读取数据模式
        if not to_predict:
            if isinstance(data, str):
                examples = read_examples_from_file(data, mode)
            else:
                examples = get_examples_from_df(data)
        else:
            examples = to_predict
            no_cache = True

        # 保存数据特征文件路径
        cached_features_file = os.path.join(
            args["cache_dir"],
            "cached_{}_{}_{}_{}_{}".format(
                mode, args["model_type"], args["max_seq_length"], self.num_labels, len(examples))
        )
        if not no_cache:
            os.makedirs(self.args["cache_dir"], exist_ok=True)

        # reprocess_input_data：不使用处理好的特征，从输入数据路径再次处理特征。
        if os.path.exists(cached_features_file) and ((not args["reprocess_input_data"] and not no_cache)
                            or (mode == "eval" and args["use_cached_eval_features"] and not no_cache)):
            with open(cached_features_file, 'rb') as f:
                features = pickle.load(f, encoding='utf-8')
            if verbose:
                logger.info(f">>> 加载缓存的特征文件 {cached_features_file}")
        else:
            if verbose:
                logger.info(f">>> 生成特征文件，未使用特征缓存文件.")
            # 生成bert类模型输入特征格式
            features = convert_examples_to_features(
                examples,
                self.labels,
                args["max_seq_length"],
                tokenizer,
                # XLNet has a CLS token at the end
                cls_token_at_end=bool(args["model_type"] in ["xlnet"]),
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=2 if args["model_type"] in ["xlnet"] else 0,
                sep_token=tokenizer.sep_token,
                # RoBERTa uses an extra separator b/w pairs of sentences,
                # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                sep_token_extra=bool(args["model_type"] in ["roberta"]),
                # PAD on the left for XLNet
                pad_on_left=bool(args["model_type"] in ["xlnet"]),
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if args["model_type"] in ["xlnet"] else 0,
                pad_token_label_id=self.pad_token_label_id,
                process_count=process_count,
                silent=args["silent"],
                use_multiprocessing=args["use_multiprocessing"],
                chunksize=args["multiprocessing_chunksize"],
            )
            if not no_cache:
                with open(cached_features_file, 'wb') as f:
                    pickle.dump(features, f)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        return dataset

    def compute_metrics(self, preds, labels, **kwargs):
        """
        计算评估指标. 根据实际需要更改.
        TODO: 修改评价指标时，修改该函数。结合实际情况直接重写，相关函数： _create_training_progress_scores
        Args:
            eval_examples: 传入样本，以输出错误预测样例。
            **kwargs: metric 函数对象. 比如传入 f1=sklearn.metrics.f1_score。函数要求接受pretict和true两个值。
        Returns:
            result: Dictionary . (Matthews correlation coefficient, tp, tn, fp, fn, ...)
            wrong: 错误预测样例
        """

        assert len(preds) == len(labels)

        extra_metrics = {}
        for metric, func in kwargs.items():
            extra_metrics[metric] = func(labels, preds)

        result = {
            "precision": precision_score(labels, preds),
            "recall": recall_score(labels, preds),
            "f1_score": f1_score(labels, preds),
            **extra_metrics,
        }

        return result

    def predict(self, to_predict, split_on_space=True):
        """
        Inference.
        Args:
            to_predict: 预测文本 list
        Returns:
            preds: list.
            model_outputs: 模型raw outputs.
        """
        device = self.device
        model = self.model
        args = self.args
        pad_token_label_id = self.pad_token_label_id

        self.model.to(self.device)

        if split_on_space:
            eval_examples = [
                InputExample(i, sentence.split(), [self.labels[0] for _ in sentence.split()])
                for i, sentence in enumerate(to_predict)
            ]
        else:
            eval_examples = [
                InputExample(i, sentence, [self.labels[0] for _ in sentence])
                for i, sentence in enumerate(to_predict)
            ]

        eval_dataset = self.load_and_cache_examples(
                None, to_predict=eval_examples, verbose=True
            )

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args["eval_batch_size"])

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        # PretrainedConfig对象的属性
        model.eval()
        all_layer_hidden_states = None
        all_embedding_outputs = None
        if self.config.output_hidden_states:
            for batch in tqdm(eval_dataloader, disable=args["silent"]):
                batch = tuple(t.to(device) for t in batch)

                with torch.no_grad():
                    inputs = self._get_inputs_dict(batch)
                    outputs = model(**inputs)
                    tmp_eval_loss, logits = outputs[:2]  # outputs： ((loss), logits, (hidden_states), (attentions))
                    embedding_outputs, layer_hidden_states = outputs[2][0], outputs[2][1:]
                    eval_loss += tmp_eval_loss.mean().item()
                # layer_hidden_states： the output of each layer， each is (batch_size, sequence_length, hidden_size)
                # embedding_outputs： the initial embedding outputs， (batch_size, sequence_length, hidden_size)
                nb_eval_steps += 1
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs["labels"].detach().cpu().numpy()
                    out_input_ids = inputs["input_ids"].detach().cpu().numpy()
                    out_attention_mask = inputs["attention_mask"].detach().cpu().numpy()
                    all_layer_hidden_states = [state.detach().cpu().numpy() for state in layer_hidden_states]
                    all_embedding_outputs = embedding_outputs.detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
                    out_input_ids = np.append(out_input_ids, inputs["input_ids"].detach().cpu().numpy(), axis=0)
                    out_attention_mask = np.append(
                        out_attention_mask, inputs["attention_mask"].detach().cpu().numpy(), axis=0,
                    )
                    all_layer_hidden_states = np.append(
                        [state.detach().cpu().numpy() for state in layer_hidden_states], axis=0
                    )
                    all_embedding_outputs = np.append(embedding_outputs.detach().cpu().numpy(), axis=0)
        else:
            # 与evaluate函数相似
            for batch in tqdm(eval_dataloader, disable=args["silent"]):
                model.eval()
                batch = tuple(t.to(device) for t in batch)

                with torch.no_grad():
                    inputs = self._get_inputs_dict(batch)
                    outputs = model(**inputs)
                    tmp_eval_loss, logits = outputs[:2]
                    eval_loss += tmp_eval_loss.mean().item()

                nb_eval_steps += 1
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs["labels"].detach().cpu().numpy()
                    out_input_ids = inputs["input_ids"].detach().cpu().numpy()
                    out_attention_mask = inputs["attention_mask"].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
                    out_input_ids = np.append(out_input_ids, inputs["input_ids"].detach().cpu().numpy(), axis=0)
                    out_attention_mask = np.append(
                        out_attention_mask, inputs["attention_mask"].detach().cpu().numpy(), axis=0,
                    )

        eval_loss = eval_loss / nb_eval_steps
        token_logits = preds
        preds = np.argmax(preds, axis=2)

        label_map = {i: label for i, label in enumerate(self.labels)}
        # 与evaluate函数相似
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        if split_on_space:
            preds = [
                [{word: preds_list[i][j]} for j, word in enumerate(sentence.split()[: len(preds_list[i])])]
                for i, sentence in enumerate(to_predict)
            ]
        else:
            preds = [
                [{word: preds_list[i][j]} for j, word in enumerate(sentence[: len(preds_list[i])])]
                for i, sentence in enumerate(to_predict)
            ]

        word_tokens = []
        for n, sentence in enumerate(to_predict):
            w_log = self._convert_tokens_to_word_logits(
                out_input_ids[n], out_label_ids[n], out_attention_mask[n], token_logits[n],
            )
            word_tokens.append(w_log)

        if split_on_space:
            model_outputs = [
                [{word: word_tokens[i][j]} for j, word in enumerate(sentence.split()[: len(preds_list[i])])]
                for i, sentence in enumerate(to_predict)
            ]
        else:
            model_outputs = [
                [{word: word_tokens[i][j]} for j, word in enumerate(sentence[: len(preds_list[i])])]
                for i, sentence in enumerate(to_predict)
            ]

        return preds, model_outputs, all_layer_hidden_states, all_embedding_outputs

    def _get_ignore_ids(self):
        ignore_ids = [
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token),
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token),
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token),
        ]
        return ignore_ids

    def _convert_tokens_to_word_logits(self, input_ids, label_ids, attention_mask, logits):
        ignore_ids = self._get_ignore_ids()

        # Remove unuseful positions
        masked_ids = input_ids[(1 == attention_mask)]
        masked_labels = label_ids[(1 == attention_mask)]
        masked_logits = logits[(1 == attention_mask)]
        for id in ignore_ids:
            masked_labels = masked_labels[(id != masked_ids)]
            masked_logits = masked_logits[(id != masked_ids)]
            masked_ids = masked_ids[(id != masked_ids)]

        # Map to word logits
        word_logits = []
        tmp = []
        for n, lab in enumerate(masked_labels):
            if lab != self.pad_token_label_id:
                if n != 0:
                    word_logits.append(tmp)
                tmp = [list(masked_logits[n])]
            else:
                tmp.append(list(masked_logits[n]))
        word_logits.append(tmp)

        return word_logits

    def _get_inputs_dict(self, batch):
        """根据model_type，处理输入数据结构
        XLM, DistilBERT and RoBERTa 没有使用 segment_ids(token_type_ids)."""
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        if self.args["model_type"] != "distilbert":
            if self.args["model_type"] in ["bert", "xlnet", "albert"]:
                inputs["token_type_ids"] = batch[2]
            else:
                inputs["token_type_ids"] = None
        return inputs

    def _create_training_progress_scores(self, **kwargs):
        """TODO: 修改评价指标时，修改该函数。结合实际情况直接重写，相关函数：compute_metrics"""
        extra_metrics = {key: [] for key in kwargs}
        training_progress_scores = {
            "global_step": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "train_loss": [],
            "eval_loss": [],
            **extra_metrics,
        }
        return training_progress_scores

    def _save_model(self, output_dir=None, optimizer=None, scheduler=None, model=None, results=None):
        """保存传入的对象参数"""
        if not output_dir:
            output_dir = self.args["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        if model and not self.args["no_save"]:
            # 多GPU训练时，DataParallel包装的模型在保存时，权值参数前面会带有module字符.
            # 为了在单卡环境下可以加载模型，需要以下操作。
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            # 可选择保存优化器状态
            if optimizer and scheduler:
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

            # 保存全局参数设置为json，方便查看
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "model_args.json"), "w", encoding='utf-8') as f:
                json.dump(self.args, f, ensure_ascii=False)

        # 保存evaluate的结果
        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w", encoding="utf-8") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

    def _load_model_args(self, input_dir):
        model_args_file = os.path.join(input_dir, "model_args.json")
        if os.path.isfile(model_args_file):
            with open(model_args_file, "r", encoding='utf-8') as f:
                model_args = json.load(f)
            return model_args
