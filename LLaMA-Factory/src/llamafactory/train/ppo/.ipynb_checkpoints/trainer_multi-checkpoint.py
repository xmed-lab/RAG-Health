# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : trainer_multi.py
# Time       ：18/3/2025 3:00 pm
# Author     ：Chuang Zhao
# version    ：python 
# Description：Multi-agent PPO的实现，仿照trainer.py; train的时候改参数记得和inference保持一致（例如温度）。
"""
import warnings
import logging
import torch.distributed as dist
from transformers.tokenization_utils_base import BatchEncoding  # 去除tokenizer warningd

# 屏蔽warnings模块的警告
warnings.filterwarnings("ignore")

# 屏蔽logging模块的所有警告（包括INFO、WARNING级别）
logging.basicConfig(level=logging.ERROR)  # 仅保留ERROR及以上级别

# 屏蔽transformers库的所有警告，尤其是Trainer相关
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)

# 针对Trainer类所在的具体模块单独设置（关键）
trainer_logger = logging.getLogger("transformers.trainer")
trainer_logger.setLevel(logging.ERROR)
trainer_logger.propagate = False  # 阻止日志向上传播

# 如果你使用的是Transformers的TrainingArguments，也可以设置
training_args_logger = logging.getLogger("transformers.training_args")
training_args_logger.setLevel(logging.ERROR)

# 导入库
import random
import math
import os
import sys
import warnings
import numpy as np
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional

import torch
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm
from transformers import GenerationConfig, Trainer, TrainerControl, TrainerState
from transformers.optimization import get_scheduler
from transformers.trainer import DEFAULT_CALLBACKS
from transformers.trainer_callback import CallbackHandler
from transformers.trainer_pt_utils import remove_dummy_checkpoint
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from trl import PPOConfig, PPOTrainer
from trl.core import PPODecorators, logprobs_from_logits
from trl.models.utils import unwrap_model_for_generation
from typing_extensions import override

from ...extras import logging
from ...extras.misc import AverageMeter, count_parameters, get_current_device, get_logits_processor
from ..callbacks import FixValueHeadModelCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from .ppo_utils import dump_layernorm, get_rewards_from_server, replace_model, restore_layernorm

### self defined
import re
import time
import json
import requests
from transformers import LogitsProcessorList, LogitsProcessor
from collections import Counter
from .instructions_template import prompt_templates
from typing import List, Dict, Any, Tuple
from collections import deque
from .agent_top import AgentTop
from .agent_low import AgentLow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .normalize_answers import *
from .utils import get_emb_model, load_graph_data, load_pickle, get_normalize_text, locate_answer

if TYPE_CHECKING:  # # 如果进行类型检查，则导入必要的库
    from datasets import Dataset
    from transformers import (
        DataCollatorWithPadding,
        PreTrainedTokenizer,
        ProcessorMixin,
        Seq2SeqTrainingArguments,
        TrainerCallback,
    )
    from trl import AutoModelForCausalLMWithValueHead

    from ...hparams import FinetuningArguments, GeneratingArguments, ModelArguments

logger = logging.get_logger(__name__)

DEFINED_CONFIG = {
    "dataset": "MIII",
    "TASK": "LOS",
    "dataset_dir": "/hpc2hdd/home/sguo349/czhaobo/RAGHealth/",
    "USE_CUDA": True,
    "GPU": "0",
    "EMB": 'E5',  # 要改一起改
    "EMB_PATH": "/hpc2hdd/home/sguo349/czhaobo/huggingface/hub/e5-v2",
    "TOPK": 1,  # 一些超参数，需要保证相似。
    "DEPTH": 2,
    "max_new_tokens": 64,
    "RATIO": 0.3,  # ratio of meta
}


def remove_punctuation(text):
    punctuation = set('.,!?;:"()[]{}-')

    return ''.join(char for char in text if char not in punctuation)


def clean_and_split(text):
    cleaned_text = remove_punctuation(text).lower()
    words = cleaned_text.split()

    return words


def calculate_match_ratio(answer, document):
    common_words = {
        "in", "on", "at", "to", "for", "with", "by", "from", "about",
        "a", "an", "the",
        "it", "they", "we", "you", "he", "she", "i", "me", "my", "mine", "ours", "us", "your", "yours", "his", "hers",
        "their", "theirs",
        "and", "or", "but", "because", "if", "then", "than", "as",
        "is", "are", "was", "were", "do", "does", "did", "have", "has", "had", "having", "be", "been", "being",
        "not", "no", "nor", "none",
        "what", "where", "when", "who", "why", "how", "which", "whom", "whose",
        ".", ",", "!", "?", ";", ":", "-", "(", ")", "[", "]", "{", "}", "\"", "'", "...", "--", "/", "\\", "|", "<",
        ">", "=", "+", "*", "&", "^", "%", "$", "#", "@", "~", "`",
        "of", "that", "this", "these", "those", "such", "there", "here", "all", "any", "both", "each", "few", "more",
        "some", "most", "other", "another", "every", "either", "neither"
    }
    answer_words = [word for word in clean_and_split(answer) if word not in common_words]
    document_words = remove_punctuation(document).lower()
    match_count = sum(1 for word in answer_words if word in document_words)
    if len(answer_words) == 0:
        return 0.0
    match_ratio = match_count / (2 * len(answer_words))

    return match_ratio


def entity_match(answer, entity_list):  # 其实他是让在精排上分布一致
    document_ratios = [(document, calculate_match_ratio(answer, document)) for document in
                       entity_list]  # 直接查看answer和召回的匹配度。
    return_binary_list = [0] * len(document_ratios)
    for i in range(len(document_ratios)):
        doc_ratio = document_ratios[i]
        if doc_ratio[1] > 0:
            return_binary_list[i] = 1
        elif doc_ratio[1] == 0:
            pass

    return return_binary_list


class AllowedTokensLogitsProcessor(LogitsProcessor):
    # 只保留允许生成的token的概率，其他的token的概率设置为负无穷
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = set(allowed_token_ids)

    def __call__(self, input_ids, scores):
        device = scores.device
        mask = torch.full(scores.shape, float('-inf'), device=device)
        mask[:, list(self.allowed_token_ids)] = 0
        scores = scores + mask

        return scores


# class EnglishOnlyProcessor(LogitsProcessor):
#     def __init__(self, tokenizer):
#         self.tokenizer = tokenizer
#         self.valid_ids = self._get_valid_ids()
        
#     def _get_valid_ids(self):
#         valid_ids = set()
#         for token, idx in self.tokenizer.get_vocab().items():
#             # 允许：字母、数字、基本标点、空格
#             if all(c.isalpha() or c.isspace() or c in ".,!?;:'\"-()" for c in token):
#                 valid_ids.add(idx)
#             # 允许特殊token
#             if token.startswith("[") and token.endswith("]"):
#                 valid_ids.add(idx)
#         return valid_ids

#     def __call__(self, input_ids, scores):
#         # 将非白名单token的概率设为负无穷
#         mask = torch.ones_like(scores) * float('-inf')
#         for valid_id in self.valid_ids:
#             mask[:, valid_id] = 0
            
#         return scores + mask

import torch
from transformers import LogitsProcessor
import string

class EnglishOnlyProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.valid_ids = self._get_valid_ids()
        
    def _get_valid_ids(self):
        """创建仅包含纯英文字符的token ID白名单"""
        valid_ids = set()
        
        # 定义允许的字符集（纯英文）
        allowed_chars = set(
            string.ascii_letters +  # a-zA-Z
            string.digits +         # 0-9
            " .,!?;:'\"-()[]{}@#$%&*+=/\\|~`^_<>"  # 标点符号和特殊字符
        )
        
        # 获取所有特殊token
        special_tokens = set(self.tokenizer.all_special_tokens)
        
        for token, idx in self.tokenizer.get_vocab().items():
            # 1. 允许所有特殊token
            if token in special_tokens:
                valid_ids.add(idx)
                continue
                
            # 2. 检查token是否仅包含允许的字符
            if all(char in allowed_chars for char in token):
                valid_ids.add(idx)
                
        return valid_ids

    def __call__(self, input_ids, scores):
        # 创建一个与scores相同大小的全-inf的mask
        mask = torch.full_like(scores, float('-inf'))
        
        # 对于每个有效id，将该位置的mask值设为0
        for valid_id in self.valid_ids:
            mask[:, valid_id] = 0
            
        # 应用mask：有效token保留原分数，无效token变为-inf
        return scores + mask


class HealthPPOTrainer(PPOTrainer, Trainer):
    def __init__(
            self,
            model_args: "ModelArguments",
            training_args: "Seq2SeqTrainingArguments",
            finetuning_args: "FinetuningArguments",
            generating_args: "GeneratingArguments",
            callbacks: Optional[list["TrainerCallback"]],
            model: "AutoModelForCausalLMWithValueHead",
            reward_model: Optional["AutoModelForCausalLMWithValueHead"],
            ref_model: Optional["AutoModelForCausalLMWithValueHead"],
            tokenizer: "PreTrainedTokenizer",
            processor: Optional["ProcessorMixin"],
            data_collator: "DataCollatorWithPadding",
            train_dataset: Optional["Dataset"] = None,
            eval_dataset: Optional["Dataset"] = None,
    ) -> None:
        if eval_dataset is not None:
            raise NotImplementedError("PPOTrainer does not support eval dataset yet.")  # 还不能实现 eval dataset

        backward_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps  # overall batchsize=16; 3* gradient* per; per* gradient *4;
        ppo_config = PPOConfig(  # several ppo configurations
            # init_kl_coef =0.05,
            model_name=model_args.model_name_or_path,
            learning_rate=training_args.learning_rate,
            mini_batch_size=training_args.per_device_train_batch_size,
            batch_size=backward_batch_size * finetuning_args.ppo_buffer_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            ppo_epochs=finetuning_args.ppo_epochs,
            max_grad_norm=training_args.max_grad_norm,
            seed=training_args.seed,
            optimize_device_cache=True,
            target=finetuning_args.ppo_target,
            use_score_scaling=finetuning_args.ppo_score_norm,
            use_score_norm=finetuning_args.ppo_score_norm,
            whiten_rewards=finetuning_args.ppo_whiten_rewards,
            accelerator_kwargs={"step_scheduler_with_optimizer": False},
            log_with=training_args.report_to[0] if training_args.report_to else None,
            project_kwargs={"logging_dir": training_args.logging_dir},
        )

        # Add deepspeed config, 这些不动
        if training_args.deepspeed_plugin is not None:
            ppo_config.accelerator_kwargs["kwargs_handlers"] = [
                DistributedDataParallelKwargs(find_unused_parameters=training_args.ddp_find_unused_parameters)
            ]
            ppo_config.accelerator_kwargs["deepspeed_plugin"] = training_args.deepspeed_plugin
            if ppo_config.log_with is not None:
                logger.warning_rank0("PPOTrainer cannot use external logger when DeepSpeed is enabled.")
                ppo_config.log_with = None

        # Create optimizer and scheduler
        if training_args.max_steps > 0:
            num_training_steps = training_args.max_steps
        else:
            total_train_batch_size = backward_batch_size * finetuning_args.ppo_buffer_size * training_args.world_size
            num_training_steps = training_args.num_train_epochs * math.ceil(
                len(train_dataset) / total_train_batch_size
            )

        optimizer = self.create_optimizer(model, training_args, finetuning_args)
        scheduler = self.create_scheduler(training_args, num_training_steps, optimizer)

        PPOTrainer.__init__(  # 不用动。
            self,
            config=ppo_config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=train_dataset,
            optimizer=optimizer,
            data_collator=data_collator,
            lr_scheduler=scheduler,
        )

        self.args = training_args  # several tuning models settings
        self.check_path()  # 更新output_dir

        self.model_args = model_args
        self.finetuning_args = finetuning_args
        self.reward_model = reward_model  # 这里为None，我们无需重新train一个reward model
        self.current_device = get_current_device()  # patch for deepspeed training

        self.generation_config = GenerationConfig(  # several generation settings
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
            **generating_args.to_dict(),
        )

        # 必要的分布式设计
        self.state = TrainerState()
        self.control = TrainerControl()
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        callbacks = DEFAULT_CALLBACKS if callbacks is None else DEFAULT_CALLBACKS + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.accelerator.unwrap_model(self.model), self.tokenizer, self.optimizer, self.lr_scheduler
        )
        if self.args.max_steps > 0:
            logger.info_rank0("max_steps is given, it will override any value given in num_train_epochs")

        self.amp_context = torch.autocast(self.current_device.type)  # 自动精度开启
        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        # self defined , 我们的场景应该只有两个batchsize
        self.batch_size_1 = self.config.batch_size

        self.metapaths_dic = load_pickle(
            DEFINED_CONFIG['dataset_dir'] + 'data/ready/meta_path.pkl')  # metapaths是一个字典列表，包含name和description
        self.metapaths_dic = {key: value for key, value in self.metapaths_dic.items() if
                              key in map(str, range(0, 10))}  # 和create_sft中保持一致。

        metapaths = [self.metapaths_dic[i]['raw-meta-path'] for i in self.metapaths_dic]
        meta_ids = list(range(len(metapaths)))  # metapath ids
        self.meta_ids = meta_ids  # list(map(str, meta_ids))

        self.ratio = DEFINED_CONFIG['RATIO']
        embedding_model = get_emb_model(DEFINED_CONFIG)
        self.max_iter = DEFINED_CONFIG['DEPTH']
        self.agent_low = AgentLow(embedding_model, TOPK=DEFINED_CONFIG['TOPK'])  # 这里不再传入llm，关于llm推理以传入function为主
        self.topk = DEFINED_CONFIG['TOPK']
        self.agent_top = AgentTop(self.metapaths_dic, self.agent_low, self.ratio, self.max_iter)  # 这里不传入llm，因为要进行分布式优化
        ## 必要的index初始化， 使用单个GPU的时候可以这么做。因为不会复制Index
        # self.index_graph() # 用多个GPU会出问题, 但是这样会大大减缓速度，我想能不能换一种retreival方式，直接以服务的形式抽取， 把graph单独搞出来
        print('custom initialize success!')


        # self.logits_processor = LogitsProcessorList([
        #     EnglishOnlyProcessor(self.tokenizer)
        # ])


        # logits train
        # # final_answer: allowed_tokens
        # if DEFINED_CONFIG['TASK'] == 'SINGLE':
        #     allowed_tokens=['A', 'B', 'C', 'D', 'incomplete', 'Answer', 'is', ':']
        #     print("Allow token for answer", allowed_tokens)

        # elif DEFINED_CONFIG['TASK'] in ['MOR', 'IHM', 'REA']:
        #     allowed_tokens=['yes', 'no', 'incomplete', 'Answer', 'is', ':']
        # elif DEFINED_CONFIG['TASK'] in ['LOS']:
        #     allowed_tokens=['1', '2', '3','4', '5','6','7', '8','9', '0', 'day', 'Answer', 'is', ':']
        # allowed_token_ids = self.tokenizer.convert_tokens_to_ids(allowed_tokens) # text对应的ID
        # eos_token_id = self.tokenizer.eos_token_id
        # allowed_token_ids.append(eos_token_id)
        # self.logits_processor = LogitsProcessorList([
        #     AllowedTokensLogitsProcessor(allowed_token_ids)
        # ])

        

    def check_path(self):
        # 检查目录是否存在
        from datetime import datetime
        if os.path.exists(self.args.output_dir):
            # 生成当前时间字符串（格式：年-月-日_时-分-秒，确保唯一性）
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # 构造新的目录路径：reward-时间
            new_output_dir = f"{self.args.output_dir}-{current_time}"
            # 更新args.output_dir
            output_dir = new_output_dir
            self.args.output_dir = output_dir
        # 确保最终的目录存在（无论是否更新过）
        os.makedirs(self.args.output_dir, exist_ok=True)

    def index_graph(self):
        # 2. 创建异构图 & text chunk
        root_to_dir = "/hpc2hdd/home/sguo349/czhaobo/RAGHealth/data/ready"  # 这个路径都是公用的
        # if not os.path.exists(os.path.join(root_to_dir, 'hetero_graph.dgl')):
        #     graph, node_data, meta_paths = create_hetero_graphs(config['KG_PATH'], root_to_dir)  # kg_data用于存储
        #     # 2. 定义元路径, 需要filter掉一些
        #     filter_list = [('drug', 'drug_effect', 'effect/phenotype'), ('drug', 'drug_drug', 'drug'),
        #                    ('disease', 'disease_phenotype_positive', 'effect/phenotype'),
        #                    ('gene/protein', 'disease_protein', 'disease')]
        #     meta_paths = [i for i in meta_paths if i not in filter_list]
        #     metapaths_dic = create_meta_path(llm, meta_paths, root_to_dir)  # kg_data用于存储
        # else:  # 存储有问题
        node_data, graph, metapaths_dic = load_graph_data(root_to_dir)
        meta_paths = [metapaths_dic[i]['raw-meta-path'] for i in metapaths_dic]
        self.agent_low.load_graph(graph)
        print('Agent Load Graph Done!')

        # 5. index 节点和边文本 150G memory~1天 for edges
        node_name_list_by_type = (
            node_data.groupby('node_type')
            .apply(lambda x: x.sort_values(by='new_index')['node_name'].tolist())
            .to_dict()
        )  # {nodetype: [node_name]} ,sort by new_index
        for node_type in node_name_list_by_type:
            chosen_lines = node_data[node_data['node_type'] == node_type]['node_name'].tolist()
            self.agent_low.index_nodes(node_type=node_type, node_ids=list(range(len(chosen_lines))),
                                       texts=chosen_lines)

        # index edges
        for path in meta_paths:
            edge_ids = graph.edges(etype=path, form='eid')  # tensor
            rel = path[1]
            src_id, tgt_id = graph.edges(etype=path)  # 获取node_id, tensor list
            src_name = np.array(node_name_list_by_type[path[0]])[src_id].tolist()
            tgt_name = np.array(node_name_list_by_type[path[2]])[tgt_id].tolist()
            src_rel_tgt = [f"{src} {rel} {tgt}" for src, tgt in zip(src_name, tgt_name)]
            self.agent_low.index_edges(edge_type=path, edge_ids=edge_ids, texts=src_rel_tgt)

        print('Agent Graph initialize Done!')

    def re_construct_format(self, reason_paths, querys, final_answers, gold_answers,
                            metapaths):  # 这其实意味着同一次reason_path只对应一个reward。
        """
        将数据转换为 LLM Factory 格式, ours
        几种测试方案
        1. 只使用query-final answer pair
        2. query-final answer pair+ 过程，均等
        3. query-final answer pair主导。子过程的reward降低。
        4. 只用PPO强化关键点，比如selection, final_reward
        """
        # 读取meta_path, 公用
        meta_descriptions = "\n".join([
            f"ID: {index}\nMeta_path: {meta['meta-path']}\n"
            for index, meta in metapaths.items()
        ])  # 和agent_top中 meta_description保持一致

        all_datas = []
        all_query_final_pairs = []

        for line, query, final_ans, gold_answer in zip(reason_paths, querys, final_answers,
                                                       gold_answers):  # 注意，这里最开始的rewrite没有添加，如果要添加，需要重新SFT。我觉得可以用follow up的能力弥补，不需要添加'; 当然也可以用querys-subqueries串起来,或者外接一个rewrite器。
            all_data = []
            rewrite_lines = []
            decision_lines = []
            sub_response_lines = []
            meta_chose_lines = []
            follow_lines = []
            terminal_lines = []
            final_answer_lines = []

            no_gold_path_lines = []

            gold_path, query, final_ans, golden_answer = line, query, final_ans, gold_answer
            reason_len = len(gold_path)

            # if gold_path == [[]]:  # 按理说不该有空的。因为这里不是sampling阶段。
            no_gold_path_data = {
                'instruction': '',
                'input': prompt_templates['complete_checking'].format(query=query, reason_history=''),  #
                'output': final_ans
            }
            no_gold_path_lines.append(json.dumps(no_gold_path_data))
            all_data.extend(no_gold_path_lines)
            if gold_path != [[]]:
                gold_path = gold_path[0]
                for index, path in enumerate(gold_path):  # [{}],
                    # path = path[0]  # 字典
                    source, subquery, subanswer, decision, final_check = path['source'], path['subquery'], path['subanswer'], path[
                        'decision'], path['final_check']
                    chosen_metapaths, kg_results, naive_results = path['chosen_metapaths'], path['chosen_entities'], \
                        path[
                            'naive_results']

                    if index == 0:  # 第一次要把rewrite加进去
                        rewrite = path['rewrite']
                        rewrite_data = {
                            'instruction': '',
                            'input': prompt_templates['rewrite'].format(query=query),  # 构建决定
                            'output': rewrite
                        }
                        rewrite_lines.append(json.dumps(rewrite_data))  # 这里是直接rewrite的

                    reason = gold_path[index - 1]['reason_history'] if index > 0 else ''
                    cur_reason, follow = path['reason_history'], path['follow']
                    # decision
                    decision_data = {
                        'instruction': '',
                        'input': prompt_templates['decide'].format(subquery=subquery, reason_history=reason),  # 构建决定
                        'output': decision
                    }
                    decision_lines.append(json.dumps(decision_data))
                    # meta_path & sub answer
                    if source == 'LLM':
                        sub_response_data = {
                            'instruction': '',
                            'input': prompt_templates['direct_answer'].format(subquery=subquery, reason_history=reason),
                            # 构建决定
                            'output': subanswer
                        }
                        sub_response_lines.append(json.dumps(sub_response_data))
                    else:
                        meta_chose_data = {
                            'instruction': '',
                            'input': prompt_templates['meta_path'].format(subquery=subquery, reason_history=reason,
                                                                          meta_path=meta_descriptions),  # 构建决定，这种都一样的部分，感觉没必要进行PPO
                            'output': str(chosen_metapaths)
                        }
                        meta_chose_lines.append(json.dumps(meta_chose_data))

                        sub_response_data = {
                            'instruction': '',
                            'input': prompt_templates['combined_prompt'].format(subquery=subquery, kg_results='',
                                                                                naive_results=naive_results),
                            # 构建决定, 这里是不是要设置为空。kg_results,不然这些retreival内容收到奖励不合适. kg_results, 要和train_multi保持一致。
                            'output': subanswer
                        }
                        sub_response_lines.append(json.dumps(sub_response_data))
                    # terminal
                    if index < len(gold_path) - 1:
                        final_answer = {
                            'instruction': '',
                            'input': prompt_templates['complete_checking'].format(query=query,
                                                                                  reason_history=cur_reason),
                            # 构建决定
                            'output': final_check # 'incomplete', 不是标准答案。
                        }
                        final_answer_lines.append(json.dumps(final_answer))  # whether deep think
                    else:
                        terminal_data = {
                            'instruction': '',
                            'input': prompt_templates['complete_checking'].format(query=query,
                                                                                  reason_history=cur_reason),
                            # 构建决定
                            'output': final_ans
                        }
                        terminal_lines.append(json.dumps(terminal_data))

                    # follow up
                    if index > 0 and index < len(gold_path) - 1:  # 只要不是唯一或者最后一个，就有follow,
                        follow_data = {
                            'instruction': '',
                            'input': prompt_templates['follow'].format(query=query, reason_history=cur_reason),  # 构建决定
                            'output': follow
                        }
                        follow_lines.append(json.dumps(follow_data))

                # 存储
                # all_data.extend(rewrite_lines)

                all_data.extend(terminal_lines) # # 这里只是为了计算answer，其他奖励正常计算
                # if random.random()< 0.3: # 对中间过程不是绝对的信任。奖励已经给了。不加过程类似于DPO
                #     all_data.extend(follow_lines)
                #     all_data.extend(decision_lines) 
                #     all_data.extend(meta_chose_lines)
                #     all_data.extend(sub_response_lines)
                #     all_data.extend(final_answer_lines) 

            all_datas.append(all_data)  # 各种json pair; [[{},{}],[{},{}]]

        # all data 本质上是所有数据的json形式，也可以返回单独的query解析。
        queries, responses = [], []  # [[], []]
        queries_len, responses_len = [], []
        for all_data in all_datas:
            query, response = [], []
            for i in all_data:
                i = json.loads(i)
                query.append(self.get_qr_messages(i['input']))  # 这里是不是有问题，不要都用user？
                response.append(i['output']) # TODO
                # response.append(self.get_qr_messages(i['output']))
            queries.extend(query)  # N query
            responses.extend(response)  # N response
            queries_len.append(len(query))
            responses_len.append(len(response))
        return all_datas, queries, responses, queries_len, responses_len  # 注意，这里是必要的。

    def get_model_output(self, messages, template=''):
        # 获得LLM回复
        message_token_with_mask = self.trans_text_to_token(messages)  # {input_token:, input_mask}
        if template == 'complete_checking':
            message_qr_inputs, message_qr_inputs_qr_answers = self.get_inputs_qr_s_g(message_token_with_mask) # 只给一个。
        else:
            message_qr_inputs, message_qr_inputs_qr_answers = self.get_inputs_qr_s_g(message_token_with_mask)  # Generate model's responses ids given queries. （tokenizer）; self.logits_processor
        # 这里的message_qr_inputs_qr_answers是一个tensor，包含了所有的token
        response_texts = []
        for i in range(len(message_qr_inputs_qr_answers)):
            # query_text = self.tokenizer.decode(message_qr_inputs[i], skip_special_tokens=True)
            response_text = self.tokenizer.decode(message_qr_inputs_qr_answers[i], skip_special_tokens=True)
            response_texts.append(response_text)
            # response_text = response_text.strip().lower()
        # print("QQQ======", response_texts) # 看看到底有没有特殊的template
        # return response_texts[0]
        return normalize_output(response_texts[0], lowercase=False)  # 目前其实都是1，不用考虑batch size的问题, 后续检查下要不要normalize,好像也不用

    def get_answer_dict(self, answers_path):  # question answer pair
        "进行question: answer检索，方便计算reward"
        print('loading pairwise data to get golden_answer dict.')
        start_time = time.time()
        with open(answers_path, 'r') as f:
            lines = f.readlines()  # all data

        end_time = time.time()
        print('time consuming: {} seconds'.format(end_time - start_time))

        question_golden_answer_dict = {}
        question_pos_path_dict = {}
        question_neg_path_dict = {}
        for line in lines:
            line = eval(line)
            gold_path, neg_path, query, golden_answer = line['golden_paths'], line['negative_paths'], line['query'], \
                line['groundtruth']
            # prediction = gold_path[-1]['answer']
            query = query.replace(" ", "")
            query = re.sub(r'[^a-zA-Z]', '', query)

            question_golden_answer_dict[query] = golden_answer  # goldne answer
            if len(gold_path) > 0:
                question_pos_path_dict[query] = gold_path
                question_neg_path_dict[query] = neg_path[0] #random.choice(neg_path)  # 为后续rank做准备
            else:  # 反之，不需要计算rank loss，没有参考意义
                question_pos_path_dict[query] = []
                question_neg_path_dict[query] = []

        return question_golden_answer_dict, question_pos_path_dict, question_neg_path_dict

    def extract_question(self, text):
        """
        Extract the question part from the given text.

        Parameters:
        text (str) -A string containing the question and other content

        Returns:
        str -Extracts the question string, or returns an empty string if not found
        """
        # 使用正则表达式提取内容
        pattern = r'<\|start_header_id\|>user<\|end_header_id\|>(.*?)<\|start_header_id\|>assistant<\|end_header_id\|>'
        match = re.search(pattern, text, re.DOTALL)  # re.DOTALL让.匹配换行符

        if match:
            question = match.group(1).strip()
            # 清理文本中的换行符和多余空格
            question = re.sub(r'\s+', ' ', question).replace('⋮', '').strip()
            return question
        else:
            print("warning: cannot find the question.")
            return ""

    def get_qr_messages(self, question):
        messages = [
            {'role': 'user', 'content': question}
        ]
        return messages

    def trans_text_to_token(self, messages_list, use_template=True):
        # 初始化一个列表，用于存储每条消息的 token ID
        input_ids_list = []
        # 真的tmd是这个问题。
        chat_template = """
        {% for message in messages %}
            <|start_header_id|>{{ message['role'] }}<|end_header_id|>
            {{ message['content'] }}
            {{ "" if loop.last else "\n" }}
        {% endfor %}
        {% if add_generation_prompt %}
            <|start_header_id|>assistant<|end_header_id|>
        {% endif %}
        """
        
        if use_template:
            # 遍历传入的消息列表
            for messages in messages_list:
                
                self.tokenizer.chat_template = chat_template # 不然不统一了。
                
                # 使用聊天模板生成 token ID，并将结果转移到 GPU
                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    add_generation_prompt=True,  # 生成一个assisant标记，提示模型该生成了。
                ).cuda()
    
                # 将生成的 token ID 添加到列表中
                input_ids_list.append(input_ids)
        else:
            for messages in messages_list:
                # 直接编码文本，不添加任何模板
                input_ids = self.tokenizer.encode(
                    messages,
                    add_special_tokens=True,  # 添加特殊标记（如BOS/EOS）
                    return_tensors="pt"
                ).cuda()
                input_ids_list.append(input_ids)

        # 去除多余维度，确保每个输入都是一维的
        input_ids_list = [input_ids.squeeze(0) for input_ids in input_ids_list]

        # 计算所有 token ID 的最大长度，以便后续填充操作
        max_length = max(input_ids.size(0) for input_ids in input_ids_list)

        # 对每个 token ID 进行填充，使用 eos_token_id 填充到最大长度
        input_ids_padded = torch.stack([
            torch.cat([input_ids.new_full((max_length - input_ids.size(0),), self.tokenizer.eos_token_id), input_ids],
                      dim=0)
            for input_ids in input_ids_list
        ], dim=0)

        # 为每个 token ID 创建对应的 attention mask，填充部分为 0，实际 token 部分为 1
        attention_masks = torch.stack([
            torch.cat([torch.zeros(max_length - input_ids.size(0), dtype=torch.long),
                       torch.ones(input_ids.size(0), dtype=torch.long)], dim=0)
            for input_ids in input_ids_list
        ], dim=0).cuda()

        # 初始化一个字典，用于存储输入和注意力掩码
        temp_batch = {}
        temp_batch["input_ids"] = input_ids_padded
        temp_batch["attention_mask"] = attention_masks

        # 返回包含 input_ids 和 attention_mask 的字典
        return temp_batch

    def ppo_train(self, resume_from_checkpoint: Optional[str] = None) -> None:  # TODO: 重点要改
        r"""Implement training loop for the PPO stage, like _inner_training_loop() in Huggingface's Trainer."""
        if resume_from_checkpoint is not None:  # 暂时不支持检查点
            raise ValueError("`resume_from_checkpoint` will be supported in the future version.")
        # # 计算总训练批量大小
        total_train_batch_size = (
                self.args.per_device_train_batch_size
                * self.args.gradient_accumulation_steps
                * self.finetuning_args.ppo_buffer_size
                * self.args.world_size
        )
        if self.args.max_steps > 0:
            num_examples = total_train_batch_size * self.args.max_steps
            num_train_epochs = sys.maxsize
            max_steps = self.args.max_steps
            steps_in_epoch = self.args.max_steps
        else:
            len_dataloader = len(self.dataloader)
            num_examples = len(self.dataset)
            num_train_epochs = self.args.num_train_epochs
            max_steps = math.ceil(num_train_epochs * len_dataloader)
            steps_in_epoch = len_dataloader

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        logger.info_rank0("***** Running training *****")  # 先打印一些必要的训练日志
        logger.info_rank0(f"  Num examples = {num_examples:,}")
        logger.info_rank0(f"  Num Epochs = {num_train_epochs:,}")
        logger.info_rank0(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        logger.info_rank0(
            f"  Total train batch size (w. parallel, buffer, distributed & accumulation) = {total_train_batch_size:,}"
        )
        logger.info_rank0(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps:,}")
        logger.info_rank0(f"  Num optimization epochs per batch = {self.finetuning_args.ppo_epochs:,}")
        logger.info_rank0(f"  Total training steps = {max_steps:,}")
        logger.info_rank0(f"  Number of trainable parameters = {count_parameters(self.model)[0]:,}")

        dataiter = iter(self.dataloader)
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        self.callback_handler.on_train_begin(self.args, self.state, self.control)


        # #### self defined file and generation control
        # LLM warm rollout
        answers_path = DEFINED_CONFIG['dataset_dir'] + 'data/{}/{}/processed/train_sft_data.jsonl'.format(
            DEFINED_CONFIG['TASK'], DEFINED_CONFIG['dataset'])
        questions_golden_answers_dict, question_pos_path_dict, question_neg_path_dict = self.get_answer_dict(
            answers_path)  # question:answer
        # kl file
        # self.config.init_kl_coef = 0.3
        kl_ctl_results_path = self.args.output_dir + '/kl_ctl.txt'
        os.makedirs(os.path.dirname(kl_ctl_results_path), exist_ok=True)
        with open(kl_ctl_results_path, 'a') as file:
            file.write('self.config.init_kl_coef: {}, self.config.target: {}, self.config.horizon: {}'.format(
                self.config.init_kl_coef, self.config.target, self.config.horizon) + '\n\n\n')

        print('self.batch_size_1: {}, self.config.mini_batch_size: {}'.format(self.batch_size_1,
                                                                              self.config.mini_batch_size))
        print('self.config.ppo_epochs: {}'.format(self.config.ppo_epochs))  # 这里为啥要重新定义这个ppo_EPOCH

        for step in tqdm(range(max_steps), disable=not self.is_local_process_zero()):
            kl_ctl_results_path = self.args.output_dir + '/kl_ctl.txt'  # 专门记录kl损失
            with open(kl_ctl_results_path, 'a') as file:
                file.write('step: {}, '.format(step) + 'self.kl_ctl.value: {}'.format(self.kl_ctl.value) + '\n')
            try:
                batch = next(dataiter)
            except StopIteration:
                dataiter = iter(self.dataloader)
                batch = next(dataiter)

            # Get inputs, 先进行rollout
            self.model.eval()
            self.tokenizer.padding_side = "right"  # change padding side
            queries, responses, rewards = [], [], []
            rewards_repeat_len = []

            torch.cuda.empty_cache()

            # 这里是batch内部处理, 这个的batch_size_1就是config.batch_size吗
            for idx in range(0, self.batch_size_1, self.config.mini_batch_size):  # 这里需要看下两个agent是不是要batch交替训练
                # get init text from mini_batch
                mini_batch = batch["input_ids"][idx: idx + self.config.mini_batch_size]  # 这里的batch是padding后的
                # Step 1: 获取question 的 text.
                init_texts = []  # 记录文本
                for sub_idx in range(self.config.mini_batch_size):
                    init_text = self.tokenizer.decode(mini_batch[sub_idx], skip_special_tokens=True)
                    init_texts.append(init_text)
                # print("XXXXX", init_text) # 还真有assissant这种东西。
                mini_batch_input_questions = []
                for text in init_texts:
                    question = self.extract_question(text)  # 获得对应的question，这里之前被加上了一点system模版
                    mini_batch_input_questions.append(question) # ['',''], 正确

                # ****************************************** Agent Flow ******************************************
                # 进入agent_run, # 需要提前获得Qwen7B的negative path {'gold_path, negative_path'} 让golden_path和Qwen7B的更近，negative path更远
                mini_batch_final_answers, mini_batch_reason_historys_path, _ = [], [], []  # 获取回复，准备计算reward
                for i in range(len(mini_batch_input_questions)):
                    final_answer, golden_path_lis, reverse_negative_path_lis = self.agent_top.predict(
                        mini_batch_input_questions[i], llm_function=self.get_model_output, topk=self.topk)
                    mini_batch_final_answers.append(final_answer)
                    mini_batch_reason_historys_path.append(golden_path_lis)  # 这里不区分是否是真的golden_path，只是思维路径。
                    # mini_batch_negative_reason_path.append(reverse_negative_path_lis)
                # print("AAAAA", golden_path_lis)


                # ****************************************** rewards ******************************************（按理说上面获得的格式应该和main一致，这里就要开始了。）
                ## get golden answers
                golden_answers, pos_paths, neg_paths = [], [], []
                for batch_i in range(len(mini_batch_input_questions)):
                    question = mini_batch_input_questions[batch_i]
                    try:
                        question = question.replace(" ", "")
                        question = re.sub(r'[^a-zA-Z]', '', question)  # 需要和之前保存的7B，保持一致
                        golden_answers.append(questions_golden_answers_dict[
                                                  question])  # 每个问题都有黄金pair answer，因为7B存了所有的数据 ,{question: answer pair}
                        pos_paths.append(
                            question_pos_path_dict[question])  # 每个问题都有黄金pair answer ,{question: golden path pair}
                        neg_paths.append(
                            question_neg_path_dict[question])  # 每个问题都有黄金pair answer ,{question: neg path pair}
                        # print('Find success exp!')
                    except KeyError:  # 多半是没改DEFAULT CONFIG的dataset路径;
                        golden_answers.append("")
                        pos_paths.append("")
                        neg_paths.append("")
                        print('KeyError: {}'.format(question))


                ####### agent_top reward, 下面算reward的时候得用path啊。
                mini_batch_agenttop_reward = self.get_atop_rewards(mini_batch_final_answers, golden_answers,
                                                                   mini_batch_reason_historys_path,
                                                                   mini_batch_input_questions)  # subquery answers, 一般都是2.除非主动进行探索。

                mini_batch_agenttop = mini_batch_agenttop_reward  # [1,2,1,1], len(mini_batch)=4
                ####### agent_low reward, 她这个只有punish, 成本函数，没有reward
                mini_batch_agentlow_reward = self.get_alow_rewards(mini_batch_final_answers, golden_answers,
                                                                   mini_batch_reason_historys_path,
                                                                   mini_batch_input_questions)  # subquery answers
                mini_batch_agentlow = mini_batch_agentlow_reward  # [1,2,1,1], len(mini_batch)=4
                ####### share answer reward, 其实他们都是同一组actor，用来不同的instructions进行区分罢了。
                mini_batch_reward = self.ashare_rewards(mini_batch_final_answers, golden_answers,
                                                        mini_batch_reason_historys_path,  # retrieval docs
                                                        mini_batch_input_questions, pos_paths,
                                                        neg_paths)  # [-2,10,7,5], len(mini_batch)=4

                mini_batch_reward_tmp = mini_batch_reward + mini_batch_agenttop + mini_batch_agentlow
                rewards.extend(mini_batch_reward_tmp)  # all rewards, [0,14,9,7], 注意这里不是一个完整的batch,当mini_bacth完了之后才是。

                
                ## get predict answers, final answers.
                for temp_id in range(len(mini_batch_final_answers)):
                    pred_ans = mini_batch_final_answers[temp_id]
                    gold_ans = golden_answers[temp_id]
                    reward_ans = mini_batch_reward_tmp[temp_id]
                    atop_reward_ans =  str(mini_batch_agenttop[temp_id])
                    alow_reward_ans = str(mini_batch_agentlow[temp_id])
                    ans_ans = str(mini_batch_reward[temp_id])
                    generator_results_path = self.args.output_dir + '/context_generator.txt'
                    with open(generator_results_path, 'a') as file:
                        file.write(pred_ans + '\t||\t' + gold_ans +'\t||\t' + str(reward_ans)+ '\t||\t' + '('+atop_reward_ans + ',' +alow_reward_ans + ','+ ans_ans + ')' + '\n')  # 存放数据

                reward_answer = mini_batch_reward.mean().item()  # 5
                reward_agenttop = mini_batch_agenttop.mean().item()  # 1
                reward_agentlow = mini_batch_agentlow.mean().item()  # 1

                reward_answer_path = self.args.output_dir + '/reward_answer.txt'  # mini_batch 内部mean_batch_reward的强度
                with open(reward_answer_path, 'a') as file:
                    file.write(str(reward_answer) + '\n')
                reward_agenttop_path = self.args.output_dir + '/reward_agenttop.txt'
                with open(reward_agenttop_path, 'a') as file:
                    file.write(str(reward_agenttop) + '\n')
                reward_agentlow_path = self.args.output_dir + '/reward_agentlow.txt'
                with open(reward_agentlow_path, 'a') as file:
                    file.write(str(reward_agentlow) + '\n')

                _, mini_query, mini_response, mini_query_len, mini_response_len = self.re_construct_format(
                    mini_batch_reason_historys_path, mini_batch_input_questions, mini_batch_final_answers,
                    golden_answers, self.metapaths_dic)
                queries.extend(mini_query)  # 这里的query是一个list, 里面是一个个的query
                responses.extend(mini_response)  # 这里的response是一个list, 里面是一个个的response
                rewards_repeat_len.extend(mini_query_len)  # 这里一个query能扩充很多字段。
                # print("Reconstruct Success !")

            rewards = torch.concatenate(rewards, dim=0)  # 这里的rewards是一个list, 里面是一个个的reward
            print("AAAAAA", self.batch_size_1, rewards_repeat_len, rewards) # 按理说应该是mini； ppo epochs为4.
            rewards = torch.repeat_interleave(rewards, repeats=torch.tensor(rewards_repeat_len))  # 赋予整个过程
            rewards = list(rewards.unbind(dim=0))

            # 这里可以为最后的answer赋权。

            # Run PPO step
            self.model.train()
            print("===============================================")
            print("===============================================")

            # if len(queries) % self.config.backward_batch_size != 0:  # 必须补全，不然ppo trainer会有问题。有些logit会为空，这是LLAMA Factory自己的bug
            #     # 计算需要补全到的目标长度
            #     target_length = ((
            #                              len(queries) // self.config.backward_batch_size) + 1) * self.config.backward_batch_size
            #     # 计算需要复制的数量
            #     num_to_duplicate = 96-len(queries) # target_length - len(queries)
            #     # 从 queries 开头复制元素进行补全
            #     queries.extend(queries[:num_to_duplicate])
            #     responses.extend(responses[:num_to_duplicate])
            #     rewards.extend(rewards[:num_to_duplicate])
            # print("Duplicated!", len(queries), self.config.backward_batch_size, num_to_duplicate) # 这里好像补全后有问题，比如step为4，那么这里backward_batch_size为32. 然后累计的query数量为64—64—64-32，然后就会卡死

            num_to_duplicate = 2*self.config.backward_batch_size-len(queries) # 少部分会涉及agent探索过程
            if num_to_duplicate > 0:
                queries.extend(queries[:num_to_duplicate])
                responses.extend(responses[:num_to_duplicate])
                rewards.extend(rewards[:num_to_duplicate])
            else:
                queries = queries[:2*self.config.backward_batch_size]
                responses = responses[:2*self.config.backward_batch_size]
                rewards = rewards[:2*self.config.backward_batch_size]
            # print("Duplicated!", len(queries), self.config.backward_batch_size)



            # num_to_duplicate = 2*self.config.backward_batch_size-len(queries) # 少部分会涉及agent探索过程
            # queries = queries[:self.config.backward_batch_size]
            # responses = responses[:self.config.backward_batch_size]
            # rewards = rewards[:self.config.backward_batch_size]
            # print("No Duplicated!")


            
            
            # print("Be-queries XXXXXXXXXX", queries[0]) # 固定前缀
            # print("Before response XXXXXXXXXX", responses[0])
                        
            batch_token_with_mask_queries = self.trans_text_to_token(queries)  # 这里注意下会不会message的角色会有影响。
            batch_token_with_mask_responses = self.trans_text_to_token(responses,use_template=False)  # 这里直接encode好像也行？
            queries, responses = self.get_inputs(batch_token_with_mask_queries,
                                                 batch_token_with_mask_responses)  # 获取对应的text结果

            # # # 进行解码, 这里只是检测看下是啥玩意
            # response_texts = []
            # query_texts = []
            # for i in range(len(responses)):
            #     response_text = self.tokenizer.decode(responses[i], skip_special_tokens=True)
            #     query_text = self.tokenizer.decode(queries[i], skip_special_tokens=True)
            #     response_texts.append(response_text)
            #     query_texts.append(query_text)

            # print("After-queries XXXXXXXXXX", query_texts[0]) # 固定前缀, 结束。
            # print("After-responses XXXXXXXXXX", response_texts[0])
            
            print("*********Start Step************")
            self.config.batch_size = len(queries)  # 这里的batch size是一个个的query和response，符合PPO trainer的诡异要求； 不会影响前面的。

            # 添加奖励归一化（批次内标准化）
            # rewards = torch.tensor(rewards)
            # reward_mean = rewards.mean()
            # reward_std = rewards.std() + 1e-8  # 避免除以0
            # rewards = (rewards - reward_mean) / reward_std  # 标准化
            # rewards = list(rewards)

            stats = self.step(queries, responses, rewards)  # 反正可以算出一个loss，gradient上升用的。
            self.config.batch_size = self.batch_size_1 # 防止前后有奇奇怪怪的变化。
            # dist.barrier()
            print("*********End Step************")

            # ensure the minimum valule of \beta is 0.05， self DEFINED
            Min_beta = 0.05
            self.kl_ctl.value = max(self.kl_ctl.value, Min_beta)  # 防止 β 过小导致 KL 惩罚失效，避免策略更新幅度过大（即使系统认为 “可以探索”，也保留最低限度的约束）。增大的话会让模型不怎么变动。他这个会自适应衰减。

            self.tokenizer.padding_side = "left"  # restore padding side
            loss_meter.update(float(stats["ppo/loss/total"]), n=len(rewards))
            reward_meter.update(torch.stack(rewards).mean().item(), n=len(rewards))

            if self.config.log_with is not None:
                try:
                    batch["query"] = self.tokenizer.batch_decode(queries, skip_special_tokens=True)
                    batch["response"] = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
                    self.log_stats(stats, batch, rewards)
                except Exception:
                    logger.warning_rank0("Failed to save stats due to unknown errors.")

            self.state.global_step += 1
            self.callback_handler.on_step_end(self.args, self.state, self.control)

            if self.is_local_process_zero() and (step + 1) % self.args.logging_steps == 0:  # 如果是主进程并且达到记录步骤
                logs = dict(
                    loss=round(loss_meter.avg, 4),
                    reward=round(reward_meter.avg, 4),
                    entropy=stats["objective/entropy"],
                    kl=stats["objective/kl"],
                    query_len=stats["tokens/queries_len_mean"],
                    response_len=stats["tokens/responses_len_mean"],
                    learning_rate=stats["ppo/learning_rate"],
                    epoch=round(step / steps_in_epoch, 2),
                )
                tqdm.write(str(logs))
                logs["step"] = step
                self.state.log_history.append(logs)
                self.callback_handler.on_log(self.args, self.state, self.control, logs)
                loss_meter.reset()
                reward_meter.reset()

            if (step + 1) % self.args.save_steps == 0:  # save checkpoint
                self.save_model(
                    os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
                )
                self.callback_handler.on_save(self.args, self.state, self.control)

            if self.control.should_epoch_stop or self.control.should_training_stop:
                break

        self.callback_handler.on_train_end(self.args, self.state, self.control)

    @override
    def create_optimizer(
            self,
            model: "AutoModelForCausalLMWithValueHead",
            training_args: "Seq2SeqTrainingArguments",
            finetuning_args: "FinetuningArguments",
    ) -> "torch.optim.Optimizer":
        # 定义优化器
        optimizer = create_custom_optimizer(model, training_args, finetuning_args)
        if optimizer is None:
            decay_params, nodecay_params = [], []
            decay_param_names = self.get_decay_parameter_names(model)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name in decay_param_names:
                        decay_params.append(param)
                    else:
                        nodecay_params.append(param)

            optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
            param_groups = [
                dict(params=nodecay_params),
                dict(params=decay_params, weight_decay=training_args.weight_decay),
            ]
            optimizer = optim_class(param_groups, **optim_kwargs)

        return optimizer

    @override
    def create_scheduler(
            self, training_args: "Seq2SeqTrainingArguments", num_training_steps: int, optimizer: "torch.optim.Optimizer"
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(training_args, num_training_steps, optimizer)
        lr_scheduler = get_scheduler(
            training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        return lr_scheduler

    # @torch.no_grad()
    # def get_inputs_qr_s_g(self, batch: dict[str, "torch.Tensor"], logits_processor=[],
    #                       max_new_tokens=DEFINED_CONFIG['max_new_tokens']) -> tuple[
    #     list["torch.Tensor"], list["torch.Tensor"]]:
    #     r"""Generate model's responses given queries. logits_processor is a list of functions to process logits 这个就是原来的trainer."""
    #     if batch["input_ids"].size(0) == 1:  # handle llama2 ppo with gradient accumulation > 1
    #         start_index = (batch["input_ids"][0] != self.tokenizer.pad_token_id).nonzero()[0].item()
    #         for k, v in batch.items():
    #             batch[k] = v[:, start_index:]

    #     with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
    #         unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)
    #         if self.model_args.upcast_layernorm:
    #             layernorm_params = dump_layernorm(unwrapped_model)

    #         # self defined logits_processor, max_new_tokens, 因为特殊吧，不然输出很难规范化
    #         if len(logits_processor) != 0:
    #             generate_output: "torch.Tensor" = unwrapped_model.generate(
    #                 generation_config=self.generation_config, logits_processor=logits_processor,
    #                 max_new_tokens=max_new_tokens, **batch
    #             )  # 这里不希望答案生成过长。
    #         elif len(logits_processor) == 0:
    #             generate_output: "torch.Tensor" = unwrapped_model.generate(
    #                 generation_config=self.generation_config, logits_processor=get_logits_processor(),
    #                 max_new_tokens=max_new_tokens, **batch
    #             )

    #         if self.model_args.upcast_layernorm:
    #             restore_layernorm(unwrapped_model, layernorm_params)

    #     query = batch["input_ids"].detach().cpu()
    #     response = generate_output[:, batch["input_ids"].size(-1):].detach().cpu()
    #     queries, responses = [], []
    #     # 这里居然要逐query处理
    #     for i in range(len(query)):
    #         query_start_index = (query[i] != self.tokenizer.pad_token_id).nonzero()[0].item()  # [2,3,4,0,0->[2,3,4]->0
    #         response_indexes = (response[i] != self.tokenizer.pad_token_id).nonzero()  # [2,3,4,0,0->[0,1,2]

    #         if len(response_indexes) == 0:  # allow empty response， 有无eos TOKEN，本意就是去除模型的padding输出。
    #             response_length = 1
    #         elif self.tokenizer.eos_token_id == self.tokenizer.pad_token_id:  # include eos token
    #             response_length = response_indexes[-1].item() + 2
    #         else:
    #             response_length = response_indexes[-1].item() + 1

    #         queries.append(query[i, query_start_index:])  # remove padding from left
    #         responses.append(response[i, :response_length])  # remove padding from right

    #     return queries, responses

    @torch.no_grad()
    def get_inputs_qr_s_g(self, batch: dict[str, "torch.Tensor"], logits_processor=[], max_new_tokens=DEFINED_CONFIG['max_new_tokens']) -> tuple[list["torch.Tensor"], list["torch.Tensor"]]:
        r"""Generate model's responses given queries. 上述logit_processor那个就是加入了allow_token，以及max_size"""
        if batch["input_ids"].size(0) == 1:  # handle llama2 ppo with gradient accumulation > 1
            start_index = (batch["input_ids"][0] != self.tokenizer.pad_token_id).nonzero()[0].item()
            for k, v in batch.items():
                batch[k] = v[:, start_index:]

        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)
            if self.model_args.upcast_layernorm:
                layernorm_params = dump_layernorm(unwrapped_model)


            if len(logits_processor) != 0:
                generate_output: "torch.Tensor" = unwrapped_model.generate(
                    generation_config=self.generation_config, logits_processor=logits_processor,
                    max_new_tokens=max_new_tokens, **batch
                )
            elif len(logits_processor) == 0:
                generate_output: "torch.Tensor" = unwrapped_model.generate(
                    generation_config=self.generation_config, logits_processor=get_logits_processor(),
                    max_new_tokens=max_new_tokens, **batch
                )
            
            # generate_output: torch.Tensor = unwrapped_model.generate(
            #     generation_config=self.generation_config, logits_processor=get_logits_processor(), **batch
            # )
            if self.model_args.upcast_layernorm:
                restore_layernorm(unwrapped_model, layernorm_params)

        query = batch["input_ids"].detach().cpu() # 这里encode过了
        response = generate_output[:, batch["input_ids"].size(-1) :].detach().cpu() # 这里生成ourput
        queries, responses = [], []
        for i in range(len(query)): # 这里居然要逐query处理
            query_start_index = (query[i] != self.tokenizer.pad_token_id).nonzero()[0].item()
            response_indexes = (response[i] != self.tokenizer.pad_token_id).nonzero()

            if len(response_indexes) == 0:  # allow empty response
                response_length = 1
            elif self.tokenizer.eos_token_id == self.tokenizer.pad_token_id:  # include eos token
                response_length = response_indexes[-1].item() + 2
            else:
                response_length = response_indexes[-1].item() + 1

            queries.append(query[i, query_start_index:])  # remove padding from left
            responses.append(response[i, :response_length])  # remove padding from right

        return queries, responses


        
    @torch.no_grad()
    def get_inputs(self, batch_query, batch_response):
        r"""获取对应的text decode结果"""
        if batch_query["input_ids"].size(0) == 1:  # handle llama2 ppo with gradient accumulation > 1
            start_index = (batch_query["input_ids"][0] != self.tokenizer.pad_token_id).nonzero()[0].item()
            for k, v in batch_query.items():
                batch_query[k] = v[:, start_index:]

        if batch_response["input_ids"].size(0) == 1:  # handle llama2 ppo with gradient accumulation > 1
            start_index = (batch_response["input_ids"][0] != self.tokenizer.pad_token_id).nonzero()[0].item()
            for k, v in batch_response.items():
                batch_response[k] = v[:, start_index:]

        query = batch_query["input_ids"].detach().cpu()
        response = batch_response["input_ids"].detach().cpu()
        queries, responses = [], []
        # 这里居然要逐query处理
        for i in range(len(query)):
            query_start_index = (query[i] != self.tokenizer.pad_token_id).nonzero()[0].item()  # [2,3,4,0,0->[2,3,4]->0
            response_indexes = (response[i] != self.tokenizer.pad_token_id).nonzero()[0].item()  # [2,3,4,0,0->[0,1,2]

            queries.append(query[i, query_start_index:])  # remove padding from left
            responses.append(response[i, response_indexes:])  # remove padding from right

        return queries, responses

    def normalize_answer_final(self, answer, gold=False):
        final_answer = answer.split('\n\n')[-1].split('Answer: ')[-1].split('The answer is: ')[-1]
        if not gold:
            final_answer = normalize_answer(final_answer)
        else:
            final_answer = lower(final_answer)
        return final_answer

    def get_answer_rank(self, golden_answers, reason_history, negative_reason_history, query):
        rewards = []
        for i in range(len(golden_answers)):
            score = 0.0
            if golden_answers[i] == 'yes' or golden_answers[i] == 'no':
                score_our = calculate_match_ratio(query[i], reason_history[
                    i])  # yes or no， 如果是yes或者no的场景，则需要看answer和question选取的doc是否一致
                score_negative = calculate_match_ratio(query[i], negative_reason_history[i])
                if score_our > score_negative:
                    score += 1.0
            else:
                score_our = calculate_match_ratio(golden_answers[i], reason_history[i])
                score_negative = calculate_match_ratio(golden_answers[i], negative_reason_history[i])
                if score_our > score_negative:
                    score += 1.0
            rewards.append(score)

        return torch.tensor(rewards).view(-1, 1)

    def get_atop_rewards(self, predict_answers, golden_answers, reason_historys, query):
        """计算 agent_top 的奖励， reward_1 [0,1], reward2 [-1,1]->[-1,2]"""
        # 假设 L 是一个常量，表示推理链的最大长度
        L = 3  # 根据具体情况调整
        rewards = []

        # 奖励 1: 计算推理链奖励, 范围[0-1]
        def calculate_reason_reward(reason_history):
            l = max(len(reason_history), L)  # 限制最长为L
            return 1 - 0.5 * (l - 1)  # 3以下都是0

        # 奖励 2: 计算路径奖励
        def calculate_path_reward(correct_ids, erroneous_ids, duplicated_ids):
            # 处理无任何路径的情况
            if len(correct_ids) == 0 and len(erroneous_ids) == 0 and len(duplicated_ids) == 0:
                return 0
            # 无正确路径时，惩罚随错误/重复数量增加而加重（而非固定-1）
            if len(correct_ids) == 0:
                return -1  # 错误惩罚更重； (len(erroneous_ids) + 0.5 * len(duplicated_ids))

            rewards = 0
            # 遍历所有正确路径（用max长度避免zip截断）
            max_len = max(len(correct_ids), len(erroneous_ids), len(duplicated_ids))
            for i in range(max_len):
                # 取对应位置元素，无则默认长度0
                corr_len = len(correct_ids[i]) if i < len(correct_ids) else 0
                error_len = len(erroneous_ids[i]) if i < len(erroneous_ids) else 0
                dupl_len = len(duplicated_ids[i]) if i < len(duplicated_ids) else 0
                # 调整权重：错误惩罚更重（-1），重复较轻（-0.3）
                reward = corr_len - 0.5 * error_len - 0.5 * dupl_len
                rewards += reward

            # 用浮点数平均，避免精度丢失；最后的除法是因为ratio:
            output = rewards / len(correct_ids) / (
                    len(self.meta_ids) * DEFINED_CONFIG['RATIO'])  # 不放大会一直过小 （这种overlap要不都乘以2得了）
            if output < 0.1:  # 输出了过小不行，如果正确数量没有错误数量/重复数量多，要惩罚
                return -0.5

            else:
                return 1 # 希望模型能学会调用RAG过程。
            return

        for reason_history in reason_historys:  # [[[{}]], [[{}]]]
            if reason_history == [[]]:
                reward = 0
                rewards.append(reward)  # 将奖励添加到列表中
                continue
            reason_history = reason_history[0]  # []
            # 假设我们从预测和真实答案中提取相关 ID，因为压根就没有外部RAG的decision
            correct_ids = []  # 从预测和真实答案中提取正确的 ID
            erroneous_ids = []  # 从预测和真实答案中提取错误的 ID
            duplicated_ids = []  # 从预测和真实答案中提取重复的 ID
            # print("AAAAAA", reason_history)
            # 示例逻辑来填充格式正确、生成错误和重复的 ID
            for one_reason in reason_history:  # 某一个deep think,# {}
                #    one_reason = one_reason[0]
                if one_reason['source'] == 'LLM':
                    continue
                else:
                    one_metapaths = one_reason['chosen_metapaths']  # 这里只用了1，2，3,可能是7B在采样时候的问题，希望能在PPO阶段增加探索
                    # 提取数字
                    # one_metapaths = re.findall(r'\d+', one_metapaths) 本身存的时候就是ID list了
                    # 转换为集合
                    set_one = set(one_metapaths)
                    # correction
                    cor = set_one & set(self.meta_ids)
                    correct_ids.append(cor)
                    # 重复的项目
                    duplicates = {item for item in set_one if one_metapaths.count(item) > 1}
                    duplicated_ids.append(list(duplicates))  # 其实重要的是这个loss。因为肯定是correct的。
                    # A 中不在 B 中的项目
                    error = set_one - set(self.meta_ids)
                    erroneous_ids.append(list(error))

            # 计算奖励
            reward_1 = calculate_reason_reward(reason_history)  # reason length，一般为1，因为一般都是1次，
            reward_2 = calculate_path_reward(correct_ids, erroneous_ids,
                                             duplicated_ids)  # 一般是1，因为correct数量一般都是多的；[[],[]], [[],[]], [[],[]]

            # 奖励 3: 可以根据具体需求添加其他奖励逻辑
            reward_3 = 0  # 这里可以添加其他奖励计算
            # 计算总奖励
            reward = reward_1 + reward_2 + reward_3
            rewards.append(reward)  # 将奖励添加到列表中

        return torch.tensor(rewards).view(-1, 1).float()  # [1,1,2,1,1]这种

    def get_alow_rewards(self, predict_answers, golden_answers, reason_historys, querys):
        """计算 agent_low 的奖励， reward[-1,1]"""
        rewards = []
        # 计算奖励
        for reason_history, golden_answer, query in zip(reason_historys, golden_answers, querys):  # 某一个deep think
            if reason_history == [[]]:
                reward = 0
                rewards.append(reward)  # 将奖励添加到列表中
                continue

            total_rewards = []
            num = 0
            reason_history = reason_history[0]  # 取第一个元素, 因为多了一个nest; neg不用。因为neg已经取0了。
            for index, one_reason in enumerate(reason_history):  # [{}]
                #    one_reason = one_reason[0]
                if one_reason['source'] == 'LLM':
                    continue
                else:
                    query, subquery, answer, kg = query, one_reason['subquery'], one_reason['subanswer'], one_reason[
                        'chosen_entities']
                    # 计算与查询的 token 重叠,保持任务特
                    if len(answer) == 0:
                        continue
                    overlap_with_query = self.token_overlap(answer, query) / len(answer) + self.token_overlap(answer,
                                                                                                              subquery) / len(
                        answer)
                    overlap_with_query = overlap_with_query * 2  # 不然太小了
                    overlap_with_evidence = self.token_overlap(answer, golden_answer) / len(answer) * 10

                    # 累加重叠数量作为奖励
                    if overlap_with_query + overlap_with_evidence < 0.15:
                        total_rewards.append(-0.5)  # 增加中间答案对最后结果如果完全没帮助
                    else:
                        total_rewards.append(0.5)
                        # total_rewards.append(overlap_with_query + overlap_with_evidence)

                    # print("CheckALow XXXXXXXX", overlap_with_query, overlap_with_evidence)
                    num += 1

            # 计算平均奖励
            average_reward = sum(total_rewards) / num if num > 0 else 0
            rewards.append(average_reward)  # 将奖励添加到列表中
        return torch.tensor(rewards).view(-1, 1).float()  # [1,0,1,1,0]一般是这种

    def ashare_rewards(self, predict_answers, golden_answers, reason_historys, querys, positive_reason_historys,
                       negative_reason_historys, eta=1.0,
                       alpha=1.0, metric_name='f1'):
        """计算共享奖励， orm reward[0,10]"""
        rewards = []

        # 计算 R_orm
        def calculate_orm_reward(predictions, goldens, metric_name=metric_name):
            # reward: metrics
            # normalized_prediction = self.normalize_answer_final(predictions) # 这里感觉有点不对啊，他拿normalize过的prediction作为最终的prediction，但是reward优化的是选项，可能会有其他乱七八糟的字符。可以打印出来看看。
            # normalized_ground_truth = self.normalize_answer_final(goldens, gold=True)
            normalized_prediction = locate_answer(predictions)
            normalized_ground_truth = goldens
            final_normalized_prediction, final_normalized_ground_truth, _ = get_normalize_text(normalized_prediction, normalized_ground_truth, DEFINED_CONFIG) # 最终检测过关就过关咯。
            
            # print("Pred========", normalized_prediction)
            # print("GT========", normalized_ground_truth)

            
            reward_metric = {"acc": 0.0, "em": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

            if final_normalized_prediction == final_normalized_ground_truth:
                if normalized_ground_truth + ' ' + normalized_ground_truth in normalized_prediction:
                    reward_metric['em'] = 1# 0.5 # 低质量答案，重复模式不要给太多分。
                else:
                    if len(normalized_prediction)<3: # 这种不用判别。
                        reward_metric['em'] = 1.0
                    else:
                        tmp_answer = normalized_prediction.split()
                        counts = Counter(tmp_answer)
                        total = len(tmp_answer)
                        is_low = any(count / total > 0.2 for count in counts.values())
                        if is_low:
                            reward_metric['em'] = 1# 0.2 # 低质量答案
                        else:
                            reward_metric['em'] = 1.0

            if normalized_ground_truth in normalized_prediction:  # or normalized_prediction in normalized_ground_truth:
                reward_metric["acc"] = 1.0

            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens) #较小的数字
            num_same = sum(common.values())

            if len(prediction_tokens) != 0:
                precision = 1.0 * num_same / len(prediction_tokens)
            else:
                precision = 0.0

            if len(ground_truth_tokens) != 0:
                recall = 1.0 * num_same / len(ground_truth_tokens)
            else:
                recall = 0.0

            if precision + recall != 0:
                f1 = (2 * precision * recall) / (precision + recall) # D D; D->reward会等于4.
            else:
                f1 = 0.0

            
            no_weight = 2 # 加权
            # if DEFINED_CONFIG['TASK']!='LOS':
            #     if final_normalized_prediction == 1:
            #         reward_metric['em'] = reward_metric['em'] * no_weight
                    
            # if final_normalized_prediction == 1: # 让他更喜欢说XX, 只适用于yes。
            #     if final_normalized_ground_truth == 1:
            #         reward_metric['em'] = reward_metric['em'] * no_weight
            #     elif final_normalized_ground_truth == 0:
            #         reward_metric['em'] = -1
                    
            reward_metric['f1'], reward_metric['precison'], reward_metric['recall'] = f1, precision, recall
            return  5*reward_metric['em'] # +2*reward_metric['f1'] # 一旦准了，奖励就会变大, 0-10，鼓励更优秀的f1 value； 5*reward_metric['f1']  for summary;

        def calculate_orm_length(predict_answer, reason_history):
            L = 4
            words = predict_answer.split()
            word_count = len(words)

            score = 0.0
            if word_count > 10:  # 鼓励越短越好，不要超级长
                score += -0.5

            # reason history
            l = min(len(reason_history), L)  # 限制最长为5
            punish = 1 - 0.5 * (l - 1)  # 线性衰减：1→0.5→0→-0.5→-1

            # reason_history_len = len(reason_history)
            # punish = 1 - abs(reason_history_len / L - 1) # 这个一般不会超过
            # punish_score = punish[reason_history_len]

            return score  # + punish

        # 计算 R_rank
        def calculate_rank_reward(reason_history, pos_reason_history, neg_reason_history):
            if reason_history == []:
                tmp = ''
            else:
                tmp = reason_history[-1]['reason_history']
            pos_similarity = self.token_overlap(tmp, pos_reason_history[-1]['reason_history'])  # 整体序列完整的history
            neg_similarity = self.token_overlap(tmp, neg_reason_history[-1]['reason_history'])
            if pos_similarity - neg_similarity > 0:
                return 0.1
            else:
                return -0.1
            # return alpha * max(0, (pos_similarity - neg_similarity)/len(pos_reason_history[0]['reason_history'])) # 如果negative越大，则应该有负奖励

        # 计算每个奖励
        for prediction, gold_answer, reason_history, positive_reason_history, negative_reason_history, query in zip(
                predict_answers, golden_answers, reason_historys, positive_reason_historys, negative_reason_historys,
                querys):
            R_orm = calculate_orm_reward(prediction, gold_answer) #+ calculate_orm_length(prediction, reason_history)  # 10*fscore-1

            if positive_reason_history:  # 有正确答案
                try:
                    reason_history = reason_history[0]  # 最后一个reason_histroy读取了全链路
                except:
                    reason_history = []
                positive_reason_history = positive_reason_history[0] # [[]]->[]
                R_rank = calculate_rank_reward(reason_history, positive_reason_history,
                                               negative_reason_history)  # 这里注意，不正确的，这里没有rank
            else:
                R_rank = 0.0
            # 计算共享奖励
            # start = time.time()
            # print("CheckxXXXXXXXXX",  calculate_orm_reward(prediction, gold_answer), calculate_orm_length(prediction, reason_history),R_orm, R_rank)
            shared_reward = eta * R_orm +  R_rank  # 结果reward + 排序reward
            rewards.append(shared_reward)  # [10*f1-score -1 -1] # [-2,10]

        return torch.tensor(rewards).view(-1, 1)

    def token_overlap(self, answer, reference):
        """计算答案与参考之间的 token 重叠数量"""
        answer_tokens = set(answer.split())
        reference_tokens = set(reference.split())
        return len(answer_tokens.intersection(reference_tokens))

    @override
    @PPODecorators.empty_device_cache()
    def batched_forward_pass(
            self,
            model: "AutoModelForCausalLMWithValueHead",
            queries: "torch.Tensor",
            responses: "torch.Tensor",
            model_inputs: dict[str, Any],
            return_logits: bool = False,
            response_masks: Optional["torch.Tensor"] = None,
    ) -> tuple["torch.Tensor", Optional["torch.Tensor"], "torch.Tensor", "torch.Tensor"]:
        r"""Calculate model outputs in multiple batches.

        Subclass and override to inject custom behavior.
        """
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(math.ceil(bs / fbs)):  # 把一个大batch分为多个小batch
            input_kwargs = {key: value[i * fbs: (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs: (i + 1) * fbs]
            response_batch = responses[i * fbs: (i + 1) * fbs]
            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs: (i + 1) * fbs]
            input_ids = input_kwargs["input_ids"]
            attention_mask = input_kwargs["attention_mask"]

            with self.amp_context:  # support bf16
                logits, _, values = model(**input_kwargs, return_dict=True, use_cache=False)

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(len(query_batch)):
                start = len(query_batch[j]) - 1
                if attention_mask[j, 0] == 0:  # offset left padding
                    start += attention_mask[j, :].nonzero()[0].item()
                end = start + len(response_batch[j])

                if response_masks is not None:
                    response_masks_batch = torch.cat((torch.zeros_like(query_batch[j]), response_masks_batch[j]))[1:]

                masks[j, :start] = 0
                masks[j, end:] = 0
                if response_masks is not None:
                    masks[j, start:end] = masks[j, start:end] * response_masks_batch[j][start:end]

            if return_logits:
                all_logits.append(logits)
            else:
                del logits

            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    @override
    def save_model(self, output_dir: Optional[str] = None) -> None:
        r"""Save model checkpoint.

        Subclass and override to inject custom behavior.
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        if self.is_fsdp_enabled or self.is_deepspeed_enabled:
            try:
                state_dict = self.accelerator.get_state_dict(self.model)  # must be called at all ranks
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
            except ValueError:
                logger.warning_rank0(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead,"
                    " use zero_to_fp32.py to recover weights"
                )
                if self.args.should_save:
                    self._save(output_dir, state_dict={})
                # remove the dummy state_dict
                remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                self.model.save_checkpoint(output_dir)

        elif self.args.should_save:
            unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)
            self._save(output_dir, state_dict=unwrapped_model.state_dict())
