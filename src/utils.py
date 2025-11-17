# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : utils.py
# Time       ：5/3/2025 9:47 am
# Author     ：any
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

from instructions_template import metapth_desc_instructions
from metrics import multilabel_metrics_fn, regression_metrics_fn, multiclass_metrics_fn, binary_metrics_fn
from metrics import qa_metrics_fn, mqa_metrics_fn, summary_metrics_fn  # 导入自定义的metrics函数
from langchain_openai import ChatOpenAI  # https://deepseek.csdn.net/67bec9ce3b685529b700a5a4.html; 第三方大模型。https://help.aliyun.com/zh/model-studio/deepseek-api?disableWebsiteRedirect=true模型设置
from langchain_core.output_parsers import StrOutputParser  # 导入字符串输出解析器
import itertools
from typing import List, Any, Callable, Optional, Dict, Tuple

# from concurrent.futures import ThreadPoolExecutor
# def get_results_parallel(futures, max_workers=100):
#     """高效方式：并行获取 futures 结果，自动过滤失败的任务"""
#
#     def fetch_result(future):
#         try:
#             return ray.get(future)
#         except Exception as e:
#             # 可选：记录失败信息
#             # print(f"Future 失败: {str(e)}")
#             return None  # 返回 None 表示失败
#
#     # 使用线程池并行处理 futures
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         results = list(executor.map(fetch_result, futures))
#     results = [res for res in results if res is not None]
#     print(f"[DEBUG] Total result batches: {len(results)}")
#
#     # 过滤掉失败的结果（即 None）
#     return results





# # 更高效的 Ray 原生实现（替代方案）， 实测差别不大， 但是数据量会变少，GPU过忙的时候
def get_results_parallel(
        futures: List[ray.ObjectRef],
        process_func: Optional[Callable[[Any], None]] = None,
        error_func: Optional[Callable[[Exception], None]] = None,
        timeout: Optional[float] = None,
        return_exceptions: bool = False
) -> List[Any]:
    """使用 ray.wait() 的更高效实现"""
    results = []
    remaining = list(futures)

    while remaining:
        ready, remaining = ray.wait(
            remaining,
            num_returns=min(len(remaining), 10),  # 每次获取10个结果
            timeout=timeout
        )

        for obj_ref in ready:
            try:
                result = ray.get(obj_ref)
                if process_func:
                    process_func(result)
                results.append(result)
            except Exception as e:
                if error_func:
                    error_func(e)
                if return_exceptions:
                    results.append(e)

    print(f"[DEBUG] Total result batches: {len(results)}")
    # print(results)
    return results


def process_batch_with_workers(
        workers: List[ray.actor.ActorHandle],
        batch_data: List[Tuple[str, str]],
        mini_batch_size: int = 2,
        max_workers: int = 100
) -> List[Any]:
    """
    批量分配数据到Ray worker并获取结果

    参数:
        workers: Ray worker列表
        batch_data: 待处理的批次数据
        batch_size: 每个worker处理的子批次大小
        max_workers: 结果获取的并行线程数

    返回:
        处理结果列表
    """
    if not workers:
        raise ValueError("Worker列表不能为空")

    futures = []
    # 循环分配数据到各个worker
    for i in range(0, len(batch_data), mini_batch_size):
        batch_to_process = batch_data[i:i + mini_batch_size]
        worker_index = i % len(workers)
        futures.append(workers[worker_index].process_batch.remote(batch_to_process))

    # 并行获取结果并过滤失败任务
    results = get_results_parallel(futures)
    return results


def get_mode(config):
    if config['TASK'] in ['MOR', 'REA' , "IHM"]:
        mode = 'binary'
        metrics = ["roc_auc", "pr_auc", "precision", "recall", 'accuracy', 'balanced_accuracy','group_binary' ]
    elif config['TASK'] in ['LOS']:
        mode = 'multiclass'
        metrics =["roc_auc_weighted_ovr", "accuracy", "cohen_kappa", "f1_weighted", 'group_multiclass']
    elif config['TASK'] in ['DIAG', 'REC','PHE']:
        mode = 'multilabel'
        metrics = ['jaccard_samples', 'f1_samples', 'pr_auc_samples', 'roc_auc_samples', 'precision_samples', 'recall_samples']
    elif config['TASK'] in ['SINGLE']:
        mode = 'qa'
        metrics = ['accuracy', 'em', 'f1_score']
    elif config['TASK'] in ['MULTI']:
        mode = 'mqa'
        metrics = ['em', 'f1_score']
    elif config['TASK'] in ['SUMMARY']:
        mode = 'summary'
        metrics = ['rouge_L', 'bleu', 'readability','sari']
    else:
        raise NotImplementedError
    return mode, metrics


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
            preds = "NA" # 为什么len(preds)可能为空，不应该啊。
        return preds, golds, no_pred_num


    elif config['TASK'] in ['DIAG', 'REC','PHE']:
        raise ValueError(f"Mode {config['TASK']} is not supported") # 暂时没实现
    else:
        raise ValueError(f"Mode {config['TASK']} is not supported")







def get_tokenizers(dataset, special_tokens=False):
    if not special_tokens:
        special_tokens = ["<pad>", "<unk>"] # 把pad取消
    feature_keys = ["conditions", "procedures", "drugs"]
    feature_tokenizers = {}
    for feature_key in feature_keys:
        feature_tokenizers[feature_key] = Tokenizer(
            tokens=dataset.get_all_tokens(key=feature_key),
            special_tokens=special_tokens,
        )
        print(feature_key, feature_tokenizers[feature_key].get_vocabulary_size())
    return feature_tokenizers


def set_random_seed(seed):
    """ 设置随机种子以确保代码的可重复性 """
    random.seed(seed)       # Python 内置的随机库
    np.random.seed(seed)    # NumPy 库
    torch.manual_seed(seed) # PyTorch 库

    # 如果您使用 CUDA，则还需要添加以下两行代码
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU



def split_by_patient(
        dataset: SampleBaseDataset,
        ratios: Union[Tuple[float, float, float], List[float]],
        train_ratio=1.0,
        seed: Optional[int] = None,
        warm_cold: bool = False,
):
    if seed is not None:
        np.random.seed(seed)
    assert sum(ratios) == 1.0, "ratios must sum to 1.0"
    patient_indx = list(dataset.patient_to_index.keys()) # 存储数据 {patientID: [index]}
    num_patients = len(patient_indx)
    np.random.shuffle(patient_indx)
    train_patient_indx = patient_indx[: int(num_patients * ratios[0])]
    np.random.seed(seed)
    np.random.shuffle(train_patient_indx)
    train_patient_indx = train_patient_indx[: int(len(train_patient_indx) * train_ratio)]
    val_patient_indx = patient_indx[
                       int(num_patients * ratios[0]): int(
                           num_patients * (ratios[0] + ratios[1]))
                       ]
    test_patient_indx = patient_indx[int(num_patients * (ratios[0] + ratios[1])):]
    train_index = list(
        chain(*[dataset.patient_to_index[i] for i in train_patient_indx])
    )
    val_index = list(chain(*[dataset.patient_to_index[i] for i in val_patient_indx]))
    test_index = list(chain(*[dataset.patient_to_index[i] for i in test_patient_indx]))

    min_length = min(len(lst) for lst in dataset.patient_to_index.values())
    print("最短列表的长度为:", min_length)

    if warm_cold:
        warm_patient_index = []
        cold_patient_index = []
        # 这里放一些东西
        for i in test_patient_indx:
            patient_index = dataset.patient_to_index[i] # lis
            if len(patient_index) > 1: # 最少是1数据来着
                warm_patient_index.extend(patient_index)
            else:
                cold_patient_index.extend(patient_index)
        if warm_cold == 'warm':
            test_dataset = torch.utils.data.Subset(dataset, warm_patient_index)
        elif warm_cold == 'cold':
            test_dataset = torch.utils.data.Subset(dataset, cold_patient_index)
    else:
        test_dataset = torch.utils.data.Subset(dataset, test_index)

    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)
    # test_dataset = torch.utils.data.Subset(dataset, test_index)
    return train_dataset, val_dataset, test_dataset


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(data, file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_path, 'wb') as f:
        data = pickle.dump(data, f)
    print("File has beeen saved to {}.".format(file_path))
    return


def get_last_visit_sample(samples):
    """提取sample中的最后一次就诊记录"""
    last_visits = {}
    for record in samples:
        patient_id = record['patient_id']
        visit_id = float(record['visit_id'])  # 将visit_id转换为整数, 变为float是一样的，之前是int
        if patient_id not in last_visits or visit_id > float(last_visits[patient_id]['visit_id']):
            last_visits[patient_id] = record
    print("Patient Number: ", len(last_visits))
    return last_visits




def create_hetero_graphs(kg_dir, root_to_dir):
    # construct异构图
    # kg_path = os.path.join(root_dir, "data_with_path.csv") # new kg
    kg_data = pd.read_csv(kg_dir + 'kg.csv') # x_index,y_index都是从0开始标注的
    node_data = pd.read_csv(kg_dir + 'nodes.csv')
    edge_data = pd.read_csv(kg_dir + 'edges.csv')

    # print("Initial KG data:")
    # print(kg_data.head())
    # print(node_data.head())

    # 根据不同的type对x_id, y_id重新编码
    # 按 node_type 分组，生成新编码, 子type内从0开始
    node_data["new_index"] = node_data.groupby("node_type").cumcount()
    x_node_data_copy = node_data[['node_id', 'new_index', 'node_type']].copy()
    x_node_data_copy.rename(columns={'new_index': 'x_new_index', 'node_type': 'x_type', 'node_id': 'x_id'}, inplace=True)
    y_node_data_copy = node_data[['node_id', 'new_index', 'node_type']].copy()
    y_node_data_copy.rename(columns={'new_index': 'y_new_index', 'node_type': 'y_type', 'node_id': 'y_id'}, inplace=True)


    # left join
    kg_data_new = pd.merge(kg_data, x_node_data_copy, on=['x_id', 'x_type']) # inner join
    kg_data = pd.merge(kg_data_new, y_node_data_copy, on=['y_id', 'y_type'])
    print("KG merge down!")
    print(kg_data.head(5))
    print('KG reindex have been done !')

    # 获取 col1、col2 和 col3 的唯一组合
    unique_combinations = kg_data[['x_type', 'relation', 'y_type']].drop_duplicates()
    # 将唯一组合转换为列表
    unique_list = unique_combinations.values.tolist()
    kg_data['meta_path'] = kg_data['x_type'] + '->' + kg_data['relation'] + '->' + kg_data['y_type']

    edges_per_relation = {}
    for rel in unique_list:
        target_meta_path = f"{rel[0]}->{rel[1]}->{rel[2]}"
        rel_df = kg_data[kg_data['meta_path'] == target_meta_path]

        # 获取源节点和目标节点
        src_nodes = rel_df['x_new_index'].tolist()
        dst_nodes = rel_df['y_new_index'].tolist()

        # 在DGL中，每种关系需要一个元组(src_type, edge_type, dst_type)作为键
        edges_per_relation[(rel[0], rel[1], rel[2])] = (torch.tensor(src_nodes), torch.tensor(dst_nodes))

    hetero_graph = dgl.heterograph(edges_per_relation)
    print("Graph is ,", hetero_graph)

    node_data.to_csv(os.path.join(root_to_dir, "node_data.csv"))
    dgl.save_graphs(os.path.join(root_to_dir, "hetero_graph.dgl"), [hetero_graph]) # 这里不知道为啥无效,重读的时候用下面的pkl
    save_pickle(edges_per_relation, os.path.join(root_to_dir, 'edges_per_relation.pkl'))
    print("Graph have been saved!")
    return hetero_graph, node_data, edges_per_relation.keys()



def create_meta_path(llm, meta_paths, root_to_dir):
    # 定义meta-path
    meta_path_dic = {}
    prompt = PromptTemplate(
        input_variables=["meta-path"],
        template=metapth_desc_instructions
    )
    chain = prompt | llm
    for index, path in enumerate(meta_paths):
        result = chain.invoke({"meta-path": path})  # , "reason_history": reason_history 不要加这个不然会爆炸
        meta_path_dic[str(index)] = {}
        meta_path_dic[str(index)]['raw-meta-path'] = path
        meta_path_dic[str(index)]['meta-path'] = path[0] + '->' + path[1] + '->' + path[2]
        meta_path_dic[str(index)]['description'] = result
    save_pickle(meta_path_dic, os.path.join(root_to_dir, 'meta_path.pkl'))
    print("Meta-paths have been saved!")
    print("Meta-paths are as follows, ", meta_path_dic)
    return meta_path_dic



def create_naive_chunks(file_path, root_to_dir):
    """读取抽取的abstract"""
    # 读取json
    with open(file_path, 'r') as file:
        data = json.load(file)

    results = []
    for key in data:
        # each entity
        entity = data[key]
        for index in range(len(entity)):
            sub_entity = entity[index]
            if 'a' in sub_entity:
                if isinstance(sub_entity['a'], dict):
                    result = ', '.join(f"{key}: {value}" for key, value in sub_entity['a'].items())
                elif isinstance(sub_entity['a'], str):
                    result = sub_entity['a']
            else:
                continue
            results.append(result)
    save_pickle(results, os.path.join(root_to_dir, 'text_chunk.pkl'))
    print("Naive chunk have been saved!")
    return results





def load_graph_data(root_to_dir):
    node_data = pd.read_csv(os.path.join(root_to_dir, 'node_data.csv'))
    # graph,_ = dgl.load_graphs(os.path.join(root_to_dir, 'hetero_graph.dgl'))
    # graph = graph[0]
    edges_per_relation = load_pickle(os.path.join(root_to_dir, 'edges_per_relation.pkl'))
    graph = dgl.heterograph(edges_per_relation)
    # print("AAAAAA", graph.ntypes)
    # print("AAAAA", graph)
    meta_path_dic = load_pickle(os.path.join(root_to_dir, 'meta_path.pkl'))
    print("load_graph done!")
    return node_data, graph, meta_path_dic

def load_text_chunk_data(root_to_dir):
    text_chunk = load_pickle(os.path.join(root_to_dir, 'text_chunk.pkl'))
    print("load_text chunk done!")
    return text_chunk

def load_user_chunk_data(root_to_dir):
    user_chunk = load_pickle(os.path.join(root_to_dir, 'user_chunk.pkl'))
    print("load_user chunk done!")
    return user_chunk

def get_metapaths_subgraph(root_dir, hetero_graph):
    # for all potential create metapaths
    data = pd.read_csv(os.path.join(root_dir, "meta_path_desc.csv"))
    # 将两列转为字典
    result_dict = data.set_index('coarse_path')['fine_path'].to_dict() # {'A-to-C':[('a','b','c'), ('a','e','c')]}
    # 提取子图, 其实就是database
    subgraphs, i = [], 1
    print("\n计算subgraph数据库:")
    for coarse_key, meta_path_list in result_dict.items():
        # 提取包含指定边类型的子图
        subgraph = dgl.edge_type_subgraph(hetero_graph, meta_path_list)
        subgraphs.append(subgraph)
        print(f"Meta-path {i + 1} 子图:")
        i += 1

    print("\n计算meta-path可达性矩阵:")
    metapath_graphs, i = [], 0
    for coarse_key, meta_path_list in result_dict.items():
        # 追踪meta-path获取可达性图
        adj = dgl.metapath_reachable_graph(hetero_graph, meta_path_list)
        metapath_graphs.append(adj)
        print(f"Meta-path {i + 1} 可达性图:")
        i += 1

    return subgraphs, metapath_graphs





def create_heterogeneous_graph():
    """创建DGL异构图（从NetworkX转换） 这个可以放到后续修改"""

    # 创建数据字典
    # 节点数据
    user_data = {
        'features': torch.randn(4, 16),  # 假设特征维度为16
        'labels': torch.tensor([0, 1, 2, 3])
    }

    item_data = {
        'features': torch.randn(3, 16),
        'labels': torch.tensor([0, 1, 2])
    }

    category_data = {
        'features': torch.randn(2, 16),
        'labels': torch.tensor([0, 1])
    }

    # 边数据
    # 用户购买商品
    u_buy_i_src = torch.tensor([0, 0, 1, 2, 3, 3])
    u_buy_i_dst = torch.tensor([0, 1, 1, 2, 0, 2])
    u_buy_i_data = {
        'weight': torch.tensor([5.0, 3.0, 2.0, 4.0, 1.0, 3.0])
    }

    # 商品属于类别
    i_in_c_src = torch.tensor([0, 1, 2])
    i_in_c_dst = torch.tensor([0, 0, 1])
    i_in_c_data = {
        'weight': torch.tensor([1.0, 1.0, 1.0])
    }

    # 用户关注用户
    u_follow_u_src = torch.tensor([0, 1, 2, 3])
    u_follow_u_dst = torch.tensor([1, 2, 3, 0])
    u_follow_u_data = {
        'weight': torch.tensor([1.0, 1.0, 1.0, 1.0])
    }

    # 构建异构图
    graph_data = {
        # 边类型的元组格式为 (源节点类型, 边类型, 目标节点类型)
        ('用户', '购买', '商品'): (u_buy_i_src, u_buy_i_dst),
        ('商品', '属于', '类别'): (i_in_c_src, i_in_c_dst),
        ('用户', '关注', '用户'): (u_follow_u_src, u_follow_u_dst)
    }

    # 节点类型到ID范围的映射
    num_nodes_dict = {
        '用户': 4,
        '商品': 3,
        '类别': 2
    }

    # 创建异构图
    g = dgl.heterograph(graph_data, num_nodes_dict)

    # 添加节点特征
    g.nodes['用户'].data['features'] = user_data['features']
    g.nodes['用户'].data['labels'] = user_data['labels']
    g.nodes['商品'].data['features'] = item_data['features']
    g.nodes['商品'].data['labels'] = item_data['labels']
    g.nodes['类别'].data['features'] = category_data['features']
    g.nodes['类别'].data['labels'] = category_data['labels']

    # 添加边特征
    g.edges['购买'].data['weight'] = u_buy_i_data['weight']
    g.edges['属于'].data['weight'] = i_in_c_data['weight']
    g.edges['关注'].data['weight'] = u_follow_u_data['weight']

    return g


# 生成节点和边的文本描述
def generate_node_text(node_type, node_id):
    """生成节点的文本描述"""
    if node_type == '用户':
        return f"用户{node_id + 1}是一个活跃用户，经常购买商品和关注其他用户。"
    elif node_type == '商品':
        items = ["手机", "电脑", "耳机"]
        return f"商品{chr(65 + node_id)}是一款热销的{items[node_id]}，深受用户喜爱。"
    elif node_type == '类别':
        categories = ["电子产品", "配件"]
        return f"类别{chr(88 + node_id)}代表{categories[node_id]}类别，包含多种商品。"
    else:
        return f"{node_type}_{node_id}"


def generate_edge_text(src_type, edge_type, dst_type, edge_id):
    """生成边的文本描述"""
    if edge_type == '购买':
        return f"用户购买了商品，购买评分为{[5, 3, 2, 4, 1, 3][edge_id]}分。"
    elif edge_type == '属于':
        return f"商品属于该类别，关联度为{1.0}。"
    elif edge_type == '关注':
        return f"用户关注了另一个用户，关注度为{1.0}。"
    else:
        return f"{src_type}_{edge_type}_{dst_type}_{edge_id}"


# import random
# import re

# def get_dropped_text(original_text, dropout_rate=0.1, datas='eICU'):
#     # 使用正则表达式更可靠地提取Disease History
#     disease_match = re.search(r"Disease History: (\[.*?\]\s*)", original_text, re.DOTALL)
#     if not disease_match:
#         return original_text  # 如果没找到，返回原始文本
    
#     # 提取并解析疾病历史列表
#     disease_str = disease_match.group(1) + ']'
    
#     try:
#         disease_list = eval(disease_str)
#     except:
#         # 处理可能的解析错误
#         return original_text
        
#     # icd9cm = InnerMap.load("ICD9CM")
#     # 随机删减疾病历史元素
#     dropped_disease = []
#     for visit in disease_list:
#         # if datas=='eICU':
#         #     dropped_visit = []
#         #     # dropped_visit = [icd9cm.lookup(code) for code in visit if random.random() > dropout_rate]
#         #     for code in visit:
#         #         if random.random() > dropout_rate:
#         #             try:
#         #                 # 尝试查找代码
#         #                 dropped_visit.append(icd9cm.lookup(code))
#         #             except (KeyError, ValueError):  # 根据实际可能的异常类型调整
#         #                 # 如果查找失败，添加"Unknown"
#         #                 dropped_visit.append("Unknown")
#         #     dropped_disease.append(dropped_visit)            
#         # else:
#         dropped_visit = [code for code in visit if random.random() < dropout_rate]
#         dropped_disease.append(dropped_visit)
            
#     # print(dropped_disease)
#     # 替换疾病历史
#     modified_text = original_text.replace(disease_str, str(dropped_disease))

    
#     # 使用正则表达式提取Procedure History
#     if datas=='eICU':
#         proc_match = re.search(r"Procedure History: (\[.*?\]\s*)", modified_text, re.DOTALL) # 因为这里之前用drug替换的
#         if not proc_match:
#             return modified_text  # 如果没找到，返回当前修改的文本
        
#         # 提取并解析手术历史列表
#         proc_str = proc_match.group(1)
#         try:
#             proc_list = eval(proc_str)
#         except:
#             # 处理可能的解析错误
#             return modified_text
        
#         # 随机删减手术历史元素
#         dropped_proc = [proc for proc in proc_list if random.random() < dropout_rate]
        
#         # 替换手术历史
#         final_text = modified_text.replace(proc_str, str(dropped_proc))
#     else:
#         proc_match = re.search(r"Procedure History: (\[.*?\]\s*)", modified_text, re.DOTALL)
#         if not proc_match:
#             return modified_text  # 如果没找到，返回当前修改的文本
        
#         # 提取并解析手术历史列表
#         proc_str = proc_match.group(1) + ']'
#         try:
#             proc_list = eval(proc_str)
#         except:
#             # 处理可能的解析错误
#             return modified_text
        
#         # 随机删减手术历史元素
#         dropped_proc = []

#         for visit in proc_list:
#             dropped_visit = [code for code in visit if random.random() < dropout_rate]
#             dropped_proc.append(dropped_visit)
    
        
#         # 替换手术历史
#         final_text = modified_text.replace(proc_str, str(dropped_proc))
        
#     return final_text # str(disease_str) + str(proc_str) # final_text



def get_dropped_text(modified_text, dropout_rate=0.3, datas='eICU'):
    # 使用正则表达式提取Procedure History
    if datas=='eICU':
        proc_match = re.search(r"Procedure History: (\[.*?\]\s*)", modified_text, re.DOTALL)
        if not proc_match:
            return modified_text  # 如果没找到，返回当前修改的文本
        
        # 提取并解析手术历史列表
        proc_str = proc_match.group(1)
        try:
            proc_list = eval(proc_str)
        except:
            # 处理可能的解析错误
            return modified_text
        
        # 随机删减手术历史元素
        dropped_proc = ''
        
        # 替换手术历史, 迫使模型关注conditions
        if random.random() < dropout_rate:
            final_text = modified_text.replace(proc_str, str(dropped_proc))
        else:
            final_text = modified_text
    else:
        proc_match = re.search(r"Procedure History: (\[.*?\]\s*)", modified_text, re.DOTALL)
        if not proc_match:
            return modified_text  # 如果没找到，返回当前修改的文本
        
        # 提取并解析手术历史列表
        proc_str = proc_match.group(1) + ']'
        try:
            proc_list = eval(proc_str)
        except:
            # 处理可能的解析错误
            return modified_text
        
        # 随机删减手术历史元素
        dropped_proc = ''

        # 替换手术历史, 迫使模型关注conditions
        if random.random() < dropout_rate:
            final_text = modified_text.replace(proc_str, str(dropped_proc))
        else:
            final_text = modified_text
        
    return final_text




def get_llm_model(config, llm_name, api_base=None, path=None, device=None):
    # 注意这里处理不好会出现ray的错误，看不出来。
    print("Use LLM model: ", llm_name)
    if device is None:
        device = int(config['GPU']) if torch.cuda.is_available() and config['USE_CUDA'] else "cpu"
    if llm_name in ['LLAMA3-1B']: # 用于调试环境, 单卡； 也可以使用Ollama部署服务。; 注意，不支持动态参数修改操作
        # 加载模型和分词器
        model_name = path # 替换为你的本地模型路径
        # 这个方便设置动态参数
        llm = HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task="text-generation",
            pipeline_kwargs={"temperature": config['run_config_dict']['temperature'],
                             "max_new_tokens": config['run_config_dict']['max_tokens'],
                             # "device": device,
                             "return_full_text":False,
                             },  # 如果有GPU，可以使用 device=0
        ).configurable_fields(
            pipeline_kwargs=ConfigurableField(
                id="pipeline_kwargs",
                name="参数",
                description="参数"
            )
        )

    elif llm_name in ['qwen-plus', 'deepseek-v3', 'qwq-plus-latest', 'qwq-32b']:
        DASHSCOPE_API_KEY = 'xxxxx'

        # 创建共享HTTP客户端（连接池）j艾苏
        http_client = httpx.Client(
            timeout=60,  # 总超时设为60秒
            limits=httpx.Limits(
                max_keepalive_connections=50,  # 保持长连接数量
                max_connections=100  # 最大连接数
            )
        )

        if llm_name.startswith('qwen'):
            llm = Tongyi(api_key=DASHSCOPE_API_KEY, model=llm_name, temperature=config['run_config_dict']['temperature'],max_tokens=config['run_config_dict']['max_tokens'],
                        request_timeout=60,
                         http_client=http_client,  # 连接池提升复用率
                         streaming=True) # 增加请求超时时间) # 好像这个只支持阿里自己的产品"qwen-plus"
        else:
            llm = ChatOpenAI(
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", # 兼容
                api_key=DASHSCOPE_API_KEY,
                model=llm_name,  # 指定模型
                temperature=config['run_config_dict']['temperature'],
                max_tokens=config['run_config_dict']['max_tokens'],
                request_timeout=60,  # 增加请求超时时间
                max_retries=3,  # 添加重试次数
                http_client=http_client, # 连接池提升复用率
                streaming=True          # 流式传输可加速首token响应
            )
            llm = llm | StrOutputParser()

        # 指定模型
    elif llm_name in ['qwen25-7B','LLAMA3-8B', 'qwen25-32B', 'deepseekr1-7B', 'meditron-7B', 'biomistral-7B', 'LLAMA3-3B', 'qwq-32B']:
        # 设置 OpenAI API 密钥和 API 基础地址
        if api_base is None:
            print("========Please check vllm service is running========")
            api_base = "http://localhost:8870/v1"  # 本地默认服务地址
            # raise ValueError("api_base is None")

        openai_api_key = "EMPTY"
        openai_api_base = api_base #"http://localhost:8080/v1"  # 本地服务地址
        llm = VLLMOpenAI(
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base,
            model_name=config['T_LLM_PATH'],
            max_tokens=config['run_config_dict']['max_tokens'],  # 设置默认值
            temperature=config['run_config_dict']['temperature'],  # 设置默认值
            streaming=True,
            # enable_prefix_caching=True,  # 启用前缀缓存
            # model_kwargs={"stop": ["."]}, 以句号结尾，会缩短。
        ).configurable_fields(
            max_tokens=ConfigurableField(
                id="max_tokens",
                name="大语言模型数量",
                description="输出token数量"
            ),
            temperature=ConfigurableField(
                id="temperature",
                name="温度",
                description="控制生成文本的随机性"
            ),
            top_p=ConfigurableField(
                id="top_p",
                name="Top P",
                description="控制生成文本的多样性"
            )
        )

    return llm




def get_emb_model(config,device=None):
    print("Use Embedding model: ", config['EMB'])
    if device is None: # device冲定义
        device = "cuda:" + config['GPU'] if torch.cuda.is_available() and config['USE_CUDA'] else "cpu"

    if config['EMB'] in ['E5', 'BGE-M3']:
        # 初始化向量化模型
        embedding_model = HuggingFaceEmbeddings(model_name=config['EMB_PATH'],model_kwargs={"device": device})
        return embedding_model
    elif config['EMB'] in ['BioMed']:
        # medical
        embedding_model = HuggingFaceEmbeddings(model_name=config['EMB_PATH'],model_kwargs={"device": device})
        return embedding_model
    elif config['EMB'] in ['GPT-4']: # 在线API
        pass


def get_atc_name(level):
    """for atc, 这里很奇怪，level为4的话和level为3的设定一致"""
    level = level + 1
    code_sys = ATC(refresh_cache=True)  # 第一次需要
    name_map = {}
    for index in code_sys.graph.nodes:
        if len(index) == level:
            name = code_sys.graph.nodes[index]['name']
            name_map[index] = name
    return name_map

def get_node_name(code_type, reverse_stand=True):
    """for ICD9CM-diag, for ICD9PROC"""
    code_sys = InnerMap.load(code_type)
    name_map = {}
    for index in code_sys.graph.nodes:
        name = code_sys.graph.nodes[index]['name']
        name_map[index] = name
    if reverse_stand:
        name_map = {key.replace('.', ''): value for key, value in name_map.items()}  # [{ATC, name}]
    return name_map

def get_stand_system(dataset):
    """返回三个编码系统，不然太慢了"""
    if dataset=='MIMIC-III':
        diag_sys = InnerMap.load("ICD9CM")
        proc_sys = InnerMap.load("ICD9PROC")
        med_sys = ATC(refresh_cache=False)
    else:
        diag_sys = InnerMap.load("ICD10CM")
        proc_sys = InnerMap.load("ICD10PROC")
        med_sys = ATC(refresh_cache=False)
    return diag_sys, proc_sys, med_sys


def copy_file(src, dst, root_to=None, test_mode=False):
    # 源文件路径和目标文件路径
    # 复制文件
    if root_to is None:
        root_to = '/hpc2hdd/home/xxxs349/xxxc/RAGHealth/LLaMA-Factory/data/'
    else:
        root_to = root_to

    # 检查目标目录是否存在，如果不存在则创建
    if not os.path.exists(root_to):
        os.makedirs(root_to, exist_ok=True)

    if test_mode:
        # 重构一下文件格式

        with open(src, 'r') as f:
            lines = f.readlines()  # all data
        pure_path_lines = []
        for line in lines:
            line = eval(line)
            query, golden_answer = line['query'], line['groundtruth']# 这里不是那么简单似乎

            pure_path_data = {
                'instruction': '',
                'input': query,  # 构建决定
                'output': golden_answer
            }
            pure_path_lines.append(json.dumps(pure_path_data))
        with open(root_to + dst, 'w') as f:
            f.write('\n'.join(pure_path_lines))
    else:
        shutil.copy(src, root_to + dst)

    print("File copied successfully.")

# def merge_json_files(output_file, *input_files):
#     merged_data = []
#
#     for file in input_files:
#         with open(file, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#             merged_data.extend(data)
#
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(merged_data, f, ensure_ascii=False, indent=4)







###### for group analysis

def generate_rare_disease(samples, threshold, path, task='DIAG', mode='code'):
    """生成稀有疾病, train中少于threshold的疾病; 这里rare disease的定义是出现code的次数"""
    # 先找到last visit
    last_visits = get_last_visit_sample(samples).values()

    disease_num = {} # 记录数据中每个疾病的数量, 本来想加label的
    disease_num_copy = {} # 这里仅记录train中见过的疾病，即评测和替换的时候都要放上
    if mode == 'code': # 会遇到困扰，因为很多病人持续的获得一些病证
        if task == 'DIAG':
            for record in last_visits:
                for disease_lis in record['conditions']:#  + [record['labels']]: # nest list
                    for disease in disease_lis:
                        if disease not in disease_num:
                            disease_num[disease] = 1
                        else:
                            disease_num[disease] += 1
            for record in last_visits:
                for disease_lis in record['conditions']  + [record['labels']]: # nest list
                    for disease in disease_lis:
                        if disease not in disease_num_copy:
                            disease_num_copy[disease] = 1
                        else:
                            disease_num_copy[disease] += 1

        elif task in ['REC', "LOS", "REA", "MOR", "IHM"]:
            for record in last_visits:
                for disease_lis in record['conditions']:
                    for disease in disease_lis:
                        if disease not in disease_num:
                            disease_num[disease] = 1
                        else:
                            disease_num[disease] += 1

    # 按数量对字典进行排序
    sorted_items = sorted(disease_num.items(), key=lambda x: x[1], reverse=False) # {'disease':16,'dsa':18 升序}
    # 计算需要选取的item数量
    num_items_to_select = int(np.ceil(threshold * len(sorted_items)))
    num_train_drl = int(np.ceil((1-threshold) * len(sorted_items)))

    # 选取前30%的item及其数量
    tail_percent_items = sorted_items[:num_items_to_select] # Figure 1 num_train_drl, 这里是前80%的数据

    rarest_percent_items = sorted_items[:num_train_drl] # 用于对齐训练，只用头部去对齐, 有点少了啊， 这里是排除了前20%的数据 (而且是训练而非全部)
    top_percent_items = sorted_items[num_items_to_select:] # 最common的数据

    file = {
        'most_common_disease': top_percent_items,
        'all_disease': sorted_items,
        'filter_drl_items': rarest_percent_items, # 过滤名单rarest
        'rare_disease': tail_percent_items # 用于替换rare
    }
    cal_top = sum([i[1] for i in top_percent_items])
    cal_tail = sum([i[1] for i in tail_percent_items])
    print("ALLDiag {} ALL interactions {}".format(len(sorted_items),sum([i[1] for i in sorted_items]))) # # rec: 2089503, top. 3887,1984264, tail:3888,3888
    print('cal top train_drl tail num  {}, top tail interaction num: {}'.format((len(top_percent_items), len(tail_percent_items)), (cal_top, cal_tail)))


    # 生成不同的group
    if task=='DIAG':
        sorted_items_copy = sorted(disease_num_copy.items(), key=lambda x: x[1], reverse=False) # {'disease':16,'dsa':18 升序}
        num_items_to_select_copy = int(np.ceil(threshold * len(sorted_items_copy)))
        num_train_drl_copy = int(np.ceil((1 - threshold) * len(sorted_items_copy)))
        tail_percent_items_copy = sorted_items_copy[:num_items_to_select_copy] # Figure 1 num_train_drl, 这里是前80%的数据
        rarest_percent_items_copy = sorted_items_copy[:num_train_drl_copy] # 用于对齐训练，只用头部去对齐, 有点少了啊， 这里是排除了前20%的数据 (而且是训练而非全部)
        top_percent_items_copy = sorted_items_copy[num_items_to_select_copy:] # 最common的数据
        sorted_items = sorted_items_copy # 很多只在label, 所以可以这么做


    group_file = {}
    percentile_ranges = [0.0, 1/3, 2/3, 1.0]#[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for start, end in zip(percentile_ranges[:-1], percentile_ranges[1:]):
        start_index = int(start * len(sorted_items)) # 这里放copy，不然真的难搞。
        end_index = int(end * len(sorted_items))
        # group_file[f"{start * 100:.0f}%-{end * 100:.0f}%"] = [k for k, _ in sorted_items[start_index:end_index]]
        group_file[f"{start}-{end}"] = [k for k, _ in sorted_items[start_index:end_index]]

    file['group_disease'] = group_file

    save_pickle(file, path + 'rare.pkl')
    print("rare disease generate done!")

    return file


def generate_rare_patient(samples, disease_group, path):
    """for rec"""
    last_visits = get_last_visit_sample(samples).values()
    # print(disease_group)
    group_patient = {}
    for record in last_visits:
        patient_id = record['patient_id']
        conditions = set(itertools.chain.from_iterable(record['conditions']))
        for key, disease_set in disease_group.items(): # 从最稀少开始
            if len(conditions & set(disease_set)) > 0:
                group_patient[patient_id] = key
                break
    save_pickle(group_patient, path + 'rare_patient.pkl')
    print("rare patient id generate done!")
    return




if __name__ == '__main__':
    from config import config
    # openai_api_key = "EMPTY"
    # openai_api_base = "http://localhost:8000/v1"  # 本地服务地址
    # llm = VLLMOpenAI(
    #     openai_api_key=openai_api_key,
    #     openai_api_base=openai_api_base,
    #     model_name='/home/xxxc/huggingface/hub/llama3-8B/',
    #     # model_kwargs={"stop": ["."]},
    # )
    # # 假设 llm 已经正确初始化
    # prompt_templates = "请介绍一下量子计算的基本原理{query}{reason_history}"
    # # print(llm(prompt_templates.format(query="量子计算", reason_history="它是基于量子力学的原理。") ))
    # query = "Chuang"
    # reason_history = "ZHao"
    # prompt = PromptTemplate(
    #     input_variables=["query", "reason_history"],
    #     template=prompt_templates
    # )
    #
    # chain = prompt | llm
    # decision = chain.invoke({"query": query, "reason_history": reason_history})  # 会先给出query，然后才给出answer
    # print(decision)


    # 检查一下那个api行不行。
    # from config import config
    # llm_name = config['T_LLM']
    # from langchain_openai import ChatOpenAI # https://deepseek.csdn.net/67bec9ce3b685529b700a5a4.html; 第三方大模型。https://help.aliyun.com/zh/model-studio/deepseek-api?disableWebsiteRedirect=true模型设置
    # from langchain_core.output_parsers import StrOutputParser  # 导入字符串输出解析器
    #
    aliyun_api_key = "sk-xxxxxx"  # 替换为你的阿里云API Key
    llm = ChatOpenAI(
        base_url="https://dashscope.aliyuncs.com/api/v1/models",
        api_key=aliyun_api_key,
        model="deepseek-v3",  # 指定模型
        temperature=1.0,
        max_tokens=4096,
        request_timeout=120,  # 增加请求超时时间
        max_retries=3  # 添加重试次数
    )
    llm = llm | StrOutputParser()

    prompt_templates = "请介绍一下量子计算的基本原理{query}{reason_history}"
    query = "Chuang"
    reason_history = "ZHao"
    prompt = PromptTemplate(
        input_variables=["query", "reason_history"],
        template=prompt_templates
    )

    chain = prompt | llm # | StrOutputParser() | StrOutputParser()
    decision = chain.invoke({"query": query, "reason_history": reason_history})  # 会先给出query，然后才给出answer
    print(decision)
