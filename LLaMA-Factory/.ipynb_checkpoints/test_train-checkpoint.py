# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : utils.py
# Time       ：5/3/2025 9:47 am
# Author     ：Chuang Zhao
# version    ：python 
# Description：
"""
import httpx
from langchain_core.runnables import ConfigurableField
# from langchain_ollama import OllamaLLM
import string
import ray

import json
######## for healthcare process ########
import os
import pickle
import random
import re
import shutil
from itertools import chain
from typing import Optional, Tuple, Union, List

import dgl
import numpy as np
import pandas as pd
import torch
from langchain_community.llms import Tongyi
from langchain_community.llms import VLLMOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from pyhealth.datasets import SampleBaseDataset
from pyhealth.medcode import InnerMap
from pyhealth.medcode.codes.atc import ATC
from pyhealth.tokenizer import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline as Pipeline

from langchain_openai import ChatOpenAI  # https://deepseek.csdn.net/67bec9ce3b685529b700a5a4.html; 第三方大模型。https://help.aliyun.com/zh/model-studio/deepseek-api?disableWebsiteRedirect=true模型设置
from langchain_core.output_parsers import StrOutputParser  # 导入字符串输出解析器
import itertools
from typing import List, Any, Callable, Optional, Dict, Tuple


def get_metrics_fn(mode: str):
    if mode == "binary":
        return binary_metrics_fn
    elif mode == "multiclass":
        return multiclass_metrics_fn
    elif mode == "multilabel":
        return multilabel_metrics_fn
    elif mode == "regression":
        return regression_metrics_fn
    elif mode == "qa":
        return qa_metrics_fn
    elif mode == "mqa":
        return mqa_metrics_fn
    elif mode == "summary":
        return summary_metrics_fn
    else:
        raise ValueError(f"Mode {mode} is not supported")


def extract_numbers(input_string):
    # 使用正则表达式查找所有数字
    numbers = re.findall(r'\d+', input_string)
    # 转换为整数列表
    return [int(num) for num in numbers]


def normalize_answer_multichoice(s):
    "多选"
    ans = re.findall("(A|B|C|D)", s)
    if len(ans) > 0:
        return ans[0].upper()
    ans = re.findall("^\s*[\"\']?(A|B|C|D)[$/,\.\"\':]", s)
    if len(ans) > 0:
        return ans[0].upper()
    ans = re.findall("^\s*(A|B|C|D) or", s)
    if len(ans) > 0:
        return ans[0].upper()
    ans = re.findall("^\s*(A|B|C|D) and", s)
    if len(ans) > 0:
        return ans[0].upper()
    ans = re.findall("[Oo]ption (A|B|C|D)", s)
    if len(ans) > 0:
        return ans[0]
    ans = re.findall(":\s*(A|B|C|D)", s)
    if len(ans) > 0:
        return ans[0].upper()
    ans = re.findall(r"\$?\\boxed\{(A|B|C|D)", s)
    if len(ans) > 0:
        return ans[0].upper()
    ans = re.findall("\*\*[Aa]nswer:?\*\*\s*(A|B|C|D)", s)
    if len(ans) > 0:
        return ans[0].upper()
    ans = re.findall("[Aa]nswer is:?\s*\{?[\"\']?(A|B|C|D)", s)
    if len(ans) > 0:
        return ans[0].upper()
    ans = re.findall("Therefore.*(A|B|C|D)", s)
    if len(ans) > 0:
        return ans[0].upper()
    ans = re.findall("-?-?>\s*\{?[\"\']?(A|B|C|D)", s)
    if len(ans) > 0:
        return ans[0].upper()
    ans = re.findall(r"is:?[\s\n]*\*?\*?(A|B|C|D)", s)
    if len(ans) > 0:
        return ans[0].upper()
    return s.strip()



def locate_answer(s):
    if s is None:
        return "NA"
    s = re.sub('\s+', ' ', s)
    groups = re.search(r"answer_choice[\"\']:\s*\{?[\"\']?(.+?)[\"\']?\s*\}", s)
    if groups:
        return groups.group(1)
    groups = re.search(r"answer[\"\']:\s*\{?[\"\']?(.+?)[\"\']?\s*\}", s)
    if groups:
        return groups.group(1)
    groups = re.search(r"answer is:?\s*\{?[\"\']?(.+?)[\"\']?\s*\}", s)
    if groups:
        return groups.group(1)
    groups = re.search(r"[Aa]nswer\*?\*?:\s*(A|B|C|D)", s)
    if groups:
        return groups.group(1)
    groups = re.search(r"is:?\s*\*?\*?(A|B|C|D)", s)
    if groups:
        return groups.group(1)
    return s.strip()


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_normalize_text(preds, golds, config):
    # 处理的是单个！单个！！！！！
    no_pred_num = 0
    if config['TASK'] in ['MOR', 'REA', 'IHM']:
        default_answer = 0
        preds = normalize_answer(preds)
        golds = normalize_answer(golds)
        dic = {'yes': 1, 'no': 0}
        scalr_gold = dic.get(golds.lower(), 0)
        if 'yes' in preds.lower(): # 和kare差不多。
            scalr_pred = 1
        elif 'no' in preds.lower():
            scalr_pred = 0
        else:
            no_pred_num = 1
            scalr_pred = default_answer

        # 逻辑可能要改
        # print("No Pred num,", no_pred_num)

        # scalr_gold = list(map(lambda x: dic.get(x.lower(), 0), golds)) # gold比较标准
        # # pred可能就会比较模糊
        # # 使用 map + lambda 实现转换
        # scalr_pred = list(map(
        #     lambda pred: 1 if "yes" in pred.lower() else (0 if "no" in pred.lower() else 0),
        #     preds
        # ))
        return scalr_pred, scalr_gold, no_pred_num
    elif config['TASK'] in ['LOS']: # 第一个数字
        default_answer = [1] +[0]*9 # 默认值, 这里非常危险
        preds = normalize_answer(preds)
        golds = normalize_answer(golds)
        # scalr_gold = [0] * 10
        scalr_gold = extract_numbers(golds)[-1] #list(map(lambda gold: extract_numbers(golds)[0]), golds) # 第一个数字
        # scalr_gold[gold_num] = 1 # 这里是为了和pred对齐
        try:
            pred_num = extract_numbers(preds)[-1]#list(map(lambda pred: extract_numbers(pred)[0]), preds), 看起来分析类都是最后给几轮
            if pred_num >9 or pred_num < 0: # 只允许0-9
                no_pred_num = 1
                scalr_pred = default_answer
            else:
                scalr_pred = [0] * 10 # 这里是为了和gold对齐
                scalr_pred[pred_num] = 1
        except: # 没有数字
            no_pred_num = 1
            scalr_pred = default_answer
        return scalr_pred, scalr_gold, no_pred_num
    elif config['TASK'] in ['MULTIPLE']: # 部分多选题或者其他的机制。
        preds = normalize_answer_multichoice(preds)
        golds = normalize_answer_multichoice(golds)
        raise ValueError(f"Mode {config['TASK']} is not supported. No multiple choice dataset.") # 暂时没实现

    elif config['TASK'] in ['SINGLE']:
        default_answer = "A"
        preds = normalize_answer_multichoice(preds)
        golds = normalize_answer_multichoice(golds)
        if len(preds) == 0:
            no_pred_num = 1
            preds = default_answer
        return preds, golds, no_pred_num

    elif config['TASK'] in ['SUMMARY']:
        preds = normalize_answer(preds)
        golds = normalize_answer(golds)
        if len(preds) == 0:
            no_pred_num = 1
            preds = "NA"
        return preds, golds, no_pred_num


    elif config['TASK'] in ['DIAG', 'REC','PHE']:
        raise ValueError(f"Mode {config['TASK']} is not supported") # 暂时没实现
    else:
        raise ValueError(f"Mode {config['TASK']} is not supported")






if __name__ == '__main__':

    DEFINED_CONFIG = {
        "dataset": "mmlu",
        "TASK": "SINGLE",
        "dataset_dir": "/hpc2hdd/home/sguo349/czhaobo/RAGHealth/",
        "USE_CUDA": True,
        "GPU": "0",
        "EMB": 'E5',  # 要改一起改
        "EMB_PATH": "/hpc2hdd/home/sguo349/czhaobo/huggingface/hub/e5-v2",
        "TOPK": 1,  # 一些超参数，需要保证相似。
        "DEPTH": 3,
        "max_new_tokens": 256,
        "RATIO": 0.3,  # ratio of meta
    }
    data_path = '/hpc2hdd/home/sguo349/czhaobo/RAGHealth/LLaMA-Factory/saves/SUMMARY/EHC/ours/lora/' + 'reward/context_generator.txt'
    with open(data_path, 'r', encoding='utf-8') as f:
        # 按行读取并处理非空行
        pairs = [line.strip().split('||') for line in f if line.strip()]
    # 过滤格式正确的条目
    valid_pairs = [(p[0].strip(), p[1].strip()) for p in pairs if len(p) == 4] # [64*4:64*5]
    print("Len of pairs", len(pairs), len(valid_pairs))
    num = 0 
    for pred, gold in valid_pairs:
        normalized_prediction = locate_answer(pred)
        normalized_ground_truth =  gold
        final_normalized_prediction, final_normalized_ground_truth, _ = get_normalize_text(normalized_prediction, normalized_ground_truth, DEFINED_CONFIG)
        if final_normalized_prediction == final_normalized_ground_truth: # SINGLE
            num +=1
        
    print("Final accuracy in train, " , num/len(valid_pairs))

    # normalized_prediction = 'C C C D B A E F G H I J K L M N O P Q R S T U V W X Y Z b c d'
    # normalized_ground_truth = 'C C C CCCCCCC'
    # final_normalized_prediction, final_normalized_ground_truth, _ = get_normalize_text(normalized_prediction, normalized_ground_truth, DEFINED_CONFIG)
    # print(final_normalized_prediction)

        
