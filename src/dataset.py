# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : dataset.py
# Time       ：5/3/2025 9:47 am
# Author     ：Any
# version    ：python 
# Description：
"""
import sys
sys.path.append("..")
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from tqdm import tqdm
import os.path
import random # 这里可能会引入一点随机性。
from utils import get_dropped_text


import dgl
from models import AgentPipeline, KARE, MedRAG, LightRAG, CoT
# from models import get_model_workers
from utils import get_last_visit_sample,save_pickle, get_node_name, get_atc_name, get_results_parallel
# from langchain_core.prompts import PromptTemplate

import yaml
import numpy as np
from pyhealth.data import Patient, Visit
from config import config
from pyhealth.datasets import SampleEHRDataset, SampleBaseDataset
from pyhealth.datasets.utils import list_nested_levels
from typing import Dict, List
from note_pre import get_note
from datetime import datetime,timedelta
from instructions_template import task_templates, prompt_templates, metapth_desc_instructions, cot_prompt, lightrag_prompt, kare_prompt, medrag_prompt
from utils import get_llm_model, get_emb_model, create_hetero_graphs, create_meta_path,create_naive_chunks, load_graph_data, load_text_chunk_data, get_stand_system, load_user_chunk_data
from utils import load_pickle
import ray
import json
# from datasets import Dataset
import torch
from torch.utils.data import DataLoader, Dataset
from langchain.schema.runnable import RunnableConfig

def get_map_system(use_config):
    dataset = use_config['DATASET']
    task = config['TASK']
    if task in ['SINGLE', 'MULTIPLE', 'SUMMARY']:
        print("No need to get map system for QA task, use the default one. ")
        return
    if dataset == 'MIII':
        name_map_diag = get_node_name('ICD9CM')
        name_map_proc = get_node_name('ICD9PROC')
        name_map_med = get_atc_name(use_config['ATCLEVEL'])
    elif dataset == 'PIC':
        name_map_diag = get_node_name('ICD10CM')
        name_map_proc = get_node_name('ICD10PROC')
        name_map_med = get_atc_name(use_config['ATCLEVEL'])
    elif dataset == 'MIV':
        name_map_diag = get_node_name('ICD9CM') # id name
        name_map_proc = get_node_name('ICD9PROC')
        name_map_diag2 = get_node_name('ICD10CM')
        name_map_proc2 = get_node_name('ICD10PROC')
        size_origin = len(name_map_diag) + len(name_map_proc) + len(name_map_diag2) + len(name_map_proc2)
        name_map_diag.update(name_map_diag2)
        name_map_proc.update(name_map_proc2)
        print("merge_size update", size_origin, len(name_map_diag)+len(name_map_proc), len(name_map_diag)+len(name_map_proc)-size_origin)

        name_map_med = get_atc_name(use_config['ATCLEVEL']) # 这里应该一致
    elif dataset == 'eICU':
        name_map_diag = get_node_name('ICD9CM') # id name
        name_map_proc = get_node_name('ICD9PROC')
        name_map_diag2 = get_node_name('ICD10CM')
        name_map_proc2 = get_node_name('ICD10PROC')
        name_map_diag.update(name_map_diag2)
        name_map_proc.update(name_map_proc2)
        name_map_med = '' # eICU的drug没有code

    feature_map_dic = {
        'conditions': name_map_diag,
        'procedures': name_map_proc,
        'drugs': name_map_med,
    }
    return feature_map_dic



def load_sft_data(json_path, template="{instruction}\n{input}\n{output}"):
    """
    加载SFT训练数据（假设JSON格式为包含instruction/input/output字段的列表）
    Example JSON结构：
    [
        {
            "instruction": "翻译以下句子",
            "input": "Hello world",
            "output": "你好世界"
        },
        ...
    ]
    """
    with open(json_path, 'r') as f:
        raw_data = json.load(f)

    formatted_data = []
    for item in raw_data:
        # 使用模板格式化数据
        prompt = template.format(
            instruction=item["instruction"],
            input=item.get("input", ""),
            output=item["output"]
        )
        formatted_data.append({"text": prompt})  # 转换为文本格式

    return Dataset.from_list(formatted_data)

def batch_encode(batch_all, feature_keys, task, max_length=None, max_num=None, map_dic=None):
    # 对每个visit取后10个，无需encode; 这里放入一些特殊的处理
    # preprocess
    # print(batch_all)
    # break
    for feature in feature_keys:
        batch = batch_all[feature]
        if max_length is not None:
            batch = [tokens[-max_length:] for tokens in batch]
        if max_num is not None:
            batch = [
                [tokens[-max_num:] for tokens in visits] for visits in batch
            ]
        # print("Origin", feature,batch)
        # transfer from ID to title

        if config['DATASET'] == 'eICU':
            if feature in ['procedures', 'drugs']:
                batch = batch
            # else: # disease, 容易出现大规模unknown
            #     replace = lambda x: map_dic[feature].get(x, 'Unknown') if not isinstance(x, list) else list(map(replace, x))
            #     batch = list(map(replace, batch))
            # batch['procedures'], batch['drugs'] = batch['drugs'], batch['procedures'] # eiCU调换，他那个文本太扯淡了
        else:    # 标准的编码
            replace = lambda x: map_dic[feature].get(x, 'Unknown') if not isinstance(x, list) else list(map(replace, x))
            batch = list(map(replace, batch))
        batch_all[feature] = batch
        
        print("New", feature, batch)

    # print("CCCCCCC", batch_all.keys()) # CCCCCCC dict_keys(['visit_id', 'patient_id', 'conditions', 'procedures', 'drugs', 'visit_id_hist', 'drugs_hist', 'conditions_raw', 'procedures_raw', 'labels'])
    batch = np.array(batch_all['labels'])
    # print(batch)
    # print("ORIGIN Label", batch)


    if task in ['PHE']:
        batch = batch
    elif task in ['LOS']:
        batch = [str(i) + 'days' for i in batch]
    elif task in ['MOR', 'REA', 'IHM']:
        batch = np.where(batch == 1, 'yes', 'no').tolist()
        # print(a)
    batch_all['labels'] = batch
    if config['DATASET']=='eICU':
        batch_all['procedures'], batch_all['drugs'] = batch_all['drugs_hist'], batch_all['procedures'] # 原来drugs，这里写错了。drugs只有最后一次。不过不要紧，用最后一次性能也差不多。
    # else:
    #     batch_all['drugs'] = batch_all['drugs_hist'] # 太尼玛大了，用最后一次吧



    # print("New Label", batch)

    batch_feature = zip(*[batch_all[feature] for feature in feature_keys + ['labels']]) # [(cond, proc, med, label)]
    return batch_feature



def create_sft_data_think_flask(dataloader, output_path, config, map_dic, task_mode='ehr', train_mode=False,run_config:RunnableConfig=RunnableConfig(llm={}), topk=1):
    # 除了COT过程外都一摸一样
    # 1. 加载LLM模型和分词器
    if not train_mode:
        with open(output_path, 'w') as json_file:  # 使用 'w' 模式写入数据
            total_batches = len(dataloader)  # 获取总批次数量（若dataloader支持）
            for index, batch in enumerate(tqdm(dataloader, desc="Processing batches", total=total_batches)):
            # for index, batch in enumerate(dataloader):
                if task_mode == 'ehr':  # EHR task任务
                    batch = batch_encode(batch, feature_keys=config['FEATURE'], task=config['TASK'],
                                         max_length=config['MAXSEQ'], max_num=config['MAXCODESEQ'], map_dic=map_dic)
                    # 创建actor实例
                    for _, (condition, procedure, medication, ground_truth) in enumerate(batch):
                        query = task_templates[config['TASK']].format(disease_info=condition, procedure_info=procedure,
                                                                      prescription_info=medication)
                        groundtruth = ground_truth

                        # 假设这是你的字典数据
                        data_dict = {
                            "think_chain": '',
                            "query": query,
                            "final_answer": '',
                            "groundtruth": groundtruth,
                            "decision": ''
                        }
                        json.dump(data_dict, json_file)
                        json_file.write('\n')  # 每个批次结果换行

                        # if index > 1:  # 仅使用部分的数据。
                        #     break

                elif task_mode in ['mqa', 'qa', 'summary']:
                    # 创建actor实例
                    input_querys, targets = batch['input'], batch['output']
                    for _, (input_query, target) in enumerate(zip(input_querys, targets)):
                        query = task_templates[task_mode].format(query=input_query)
                        groundtruth = target

                        # 假设这是你的字典数据
                        data_dict = {
                            "think_chain": '',
                            "query": query,
                            "final_answer":'',
                            "groundtruth": groundtruth,
                            "decision": ''
                        }
                        json.dump(data_dict, json_file)
                        json_file.write('\n')  # 每个批次结果换行

                    # if index > 1:  # 仅使用部分的数据。
                    #     break

        print("Test. You have finished in the pipeline!")
        return
    else:
        llm = None
        embedding_model = None

        # Step 1. data
        # 2. 创建异构图 & text chunk， 这些都是task & dataset共享的。
        root_to_dir = "/hpc2hdd/home/xxxs349/xxxc/RAGHealth/data/ready" # 这个路径都是公用的
        if not os.path.exists(os.path.join(root_to_dir, 'hetero_graph.dgl')):
            graph, node_data, meta_paths = create_hetero_graphs(config['KG_PATH'], root_to_dir) # kg_data用于存储
            # 2. 定义元路径, 需要filter掉一些
            filter_list = [('drug', 'drug_effect', 'effect/phenotype'), ('drug', 'drug_drug', 'drug'), ('disease', 'disease_phenotype_positive', 'effect/phenotype'),('gene/protein', 'disease_protein', 'disease')]
            meta_paths = [i for i in meta_paths if i not in filter_list]
            metapaths_dic = create_meta_path(llm, meta_paths, root_to_dir) # kg_data用于存储
        else: # 存储有问题
            node_data, graph, metapaths_dic = load_graph_data(root_to_dir)
            meta_paths = [metapaths_dic[i]['raw-meta-path'] for i in metapaths_dic]

        if not os.path.exists(os.path.join(root_to_dir, 'text_chunk.pkl')): # 功能已经移到rag_flask中
            # 耗时比较大，不建议和上面合并
            text_chunks = create_naive_chunks(config['TEXT_PATH'], root_to_dir)
        else:
            text_chunks = load_text_chunk_data(root_to_dir)

        if not os.path.exists(os.path.join(root_to_dir, 'user_chunk.pkl')) and train_mode:  # 功能已经移到rag_flask中
            # 耗时比较大，不建议和上面合并
            user_chunks = create_user_chunks(dataloader, root_to_dir, task_mode, map_dic) # 只用train_loader
        else:
            user_chunks = load_user_chunk_data(root_to_dir)

        print('Agent Done! No need load graph. But you need to run the flask server!')

        # Step 2. create model
        gpu_count = torch.cuda.device_count()

        ray.init(num_gpus=gpu_count) # 用4个gpu，供所有actor可见， 最好和batch相同或者可以被batch整除 config['BATCH']
        # # 创建类的实例, 不传入gpus则默认在CPU上面运行，
        if config['MODEL'] == 'KARE':
            ray_workers = [
                KARE.remote(llm, embedding_model, meta_paths, topk=config['TOPK'], config=config) for i in range(gpu_count)]
        elif config['MODEL'] == 'MedRAG':
            ray_workers = [
                MedRAG.remote(llm, embedding_model,meta_paths, topk=config['TOPK'], config=config) for i in range(gpu_count)]
        elif config['MODEL'] == 'LightRAG':
            ray_workers = [
                LightRAG.remote(llm, embedding_model,meta_paths, topk=config['TOPK'], config=config) for i in range(gpu_count)]
        elif config['MODEL'] == 'CoT':
            ray_workers = [
                CoT.remote(llm, embedding_model, meta_paths, topk=config['TOPK'], config=config) for i in
                range(gpu_count)]
            # ray_workers = [
            #     CoT.remote(llm, embedding_model,meta_paths, topk=config['TOPK'], config=config) for i in range(config['BATCH'])]
        else:
            raise ValueError("Invalid model name. Choose from ['KARE', 'MedRAG', 'LightRAG', 'CoT']")
        print('Model Done! Please confirm the ray workers are created successfully. If not, please check the GPU setting.')



        # Step 3. continue the pipeline
        # 3. 运行 Pipeline
        with open(output_path, 'w', buffering=1) as json_file:  # 使用 'w' 模式写入数据l ,  (大数据中间会断)
            total_batches = len(dataloader)  # 获取总批次数量（若dataloader支持）
            num = 0
            for index, batch in enumerate(tqdm(dataloader, desc="Processing batches", total=total_batches)):
            # for index, batch in enumerate(dataloader):
                if task_mode == 'ehr': # EHR task任务
                    batch = batch_encode(batch, feature_keys=config['FEATURE'], task=config['TASK'], max_length=config['MAXSEQ'], max_num=config['MAXCODESEQ'], map_dic=map_dic)
                    # # 创建actor实例,
                    # futures = [ray_workers[inde].run.remote(
                    #     query=task_templates[config['TASK']].format(disease_info=condition, procedure_info=procedure,
                    #                                                 prescription_info=medication),
                    #     ground_truth=ground_truth)
                    #            for inde, (condition, procedure, medication, ground_truth) in enumerate(batch)]
                    futures = []
                    batch = list(batch)
                    mini_batch_size = config['BATCH'] // gpu_count
                    for i in range(0, len(batch), mini_batch_size):
                        # 提取mini-batch并准备输入
                        mini_batch = batch[i:i + mini_batch_size]
                        inputs = [
                            (task_templates[config['TASK']].format(
                                disease_info=cond,
                                procedure_info=proc,
                                prescription_info=med
                            ), ground_truth)
                            for cond, proc, med, ground_truth in mini_batch
                        ]
                        inputs, ground_truths = zip(*inputs)


                        # 分配给worker（循环使用）
                        worker_idx = i // mini_batch_size % len(ray_workers)
                        futures.append(ray_workers[worker_idx].batch_run.remote(inputs, ground_truths))
                elif task_mode in ['mqa','qa','summary']:
                     # 创建actor实例
                     # input_querys, targets = batch['input'], batch['output']
                     # futures = [ray_workers[inde//6].run.remote(
                     #     query=task_templates[task_mode].format(query=input_query),
                     #     ground_truth=target, run_config=run_config, topk=topk)
                     #     for inde, (input_query, target) in enumerate(zip(input_querys, targets))]
                    futures = []
                    input_querys, targets = batch['input'], batch['output']
                    # print("AAAAA", len(input_querys))
                    mini_batch_size = config['BATCH'] // gpu_count
                    for i in range(0, len(input_querys), mini_batch_size): # 这里tmd千万别搞错了，不然数据会少
                        # 提取mini-batch并准备输入
                        mini_batch_querys = input_querys[i:i + mini_batch_size]
                        mini_batch_targets = targets[i:i + mini_batch_size]
                        mini_batch_querys = [task_templates[task_mode].format(query=input_query) for input_query in mini_batch_querys]

                        # 分配给worker（循环使用）
                        worker_idx = i // mini_batch_size % len(ray_workers)
                        futures.append(ray_workers[worker_idx].batch_run.remote(mini_batch_querys, mini_batch_targets))

                # print("AAAAAAA", len(futures))
                # 每个actor进行action
                results = get_results_parallel(futures) # 加快速度，但是可能缺失。
                # 计算results的len, results是nest list

                for mini_batch_data in results:

                    for batch_data in mini_batch_data:
                        query, think_chain, final_answer, groundtruth, decision = batch_data # [{}], [[{}],[{}]], '', ''
                        # 假设这是你的字典数据
                        data_dict = {
                            "think_chain": think_chain,
                            "query": query,
                            "final_answer":final_answer,
                            "groundtruth": groundtruth,
                            "decision": decision
                        }
                        json.dump(data_dict, json_file)
                        json_file.write('\n')  # 每个批次结果换行

                        num += 1
                if index %500== 0: # 仅使用部分的数据。
                    json_file.flush()

        print("Train. You have finished in the pipeline!")
        print("Total samples: ", num)



def re_construct_format_think(path, to_path, train_mode=True,config=None):
    """
    将数据转换为 LLM Factory 格式, ours
    """
    with open(path, 'r') as f:
        lines = f.readlines() # all data
    if train_mode:
        if config['MODEL'] == 'CoT':
            print("CoT model, no need to reconstruct the data, just use the original data.")
            decision_lines = []
            rag_lines = []
            wrong_lines = [] # LLM warm start没有回答正确直接记录答案

            pure_path_lines = []


            for line in lines:
                line = eval(line)
                think_chain, query, golden_answer, final_answer, decision = line['think_chain'], line['query'], line['groundtruth'], line['final_answer'], line['decision'] # 这里不是那么简单似乎

                pure_path_data = {
                    'instruction': '',
                    'input': query,  # 构建决定
                    'output': golden_answer
                }

                if final_answer =='True':
                    # 构建decision
                    decision_data = {
                        'instruction': '',
                        'input': cot_prompt['decide'].format(subquery=query),  # 构建决定
                        'output': decision
                    }
                    decision_lines.append(json.dumps(decision_data))

                    # 构建RAG
                    rag_data = {
                        'instruction': '',
                        'input': cot_prompt['combined_prompt'].format(subquery=query, kg_results=think_chain["kg_results"], naive_results=think_chain['naive_results']),  # 构建决定
                        'output': golden_answer
                    }
                    rag_lines.append(json.dumps(rag_data))
                else:
                    wrong_data = {
                        'instruction': '',
                        'input': query,  # 构建决定
                        'output': golden_answer
                    }
                    wrong_lines.append(json.dumps(wrong_data))

                pure_path_lines.append(json.dumps(pure_path_data))


            # 存储
            with open(os.path.join(to_path, '{}_pure_path.jsonl'.format(config['MODEL'])), 'w') as f:
                f.write('\n'.join(pure_path_lines))

            with open(os.path.join(to_path, '{}_decision.jsonl'.format(config['MODEL'])), 'w') as f:
                f.write('\n'.join(decision_lines))

            with open(os.path.join(to_path, '{}_rag.jsonl'.format(config['MODEL'])), 'w') as f:
                f.write('\n'.join(rag_lines))

            with open(os.path.join(to_path, '{}_wrong.jsonl'.format(config['MODEL'])), 'w') as f:
                f.write('\n'.join(rag_lines))

            all_data = []
            all_data.extend(decision_lines)
            all_data.extend(rag_lines)
            all_data.extend(wrong_lines)
            with open(os.path.join(to_path, '{}_all_data.jsonl'.format(config['MODEL'])), 'w') as f: # 包含所有中间步骤
                f.write('\n'.join(all_data))
        elif config['MODEL'] == 'LightRAG':
            print("RAG model, no need to reconstruct the data, just use the original data.")
            rag_lines = []
            entity_lines = []
            concept_lines = []
            wrong_lines = []  # LLM warm start没有回答正确直接记录答案
            pure_path_lines = []

            for line in lines:
                line = eval(line)
                think_chain, query, golden_answer, final_answer, decision = line['think_chain'], line['query'], line[
                    'groundtruth'], line['final_answer'], line['decision']  # 这里不是那么简单似乎

                pure_path_data = {
                    'instruction': '',
                    'input': query,  # 构建决定
                    'output': golden_answer
                }

                if final_answer == 'True':
                    # entity
                    entity_data = {
                        'instruction': '',
                        'input': lightrag_prompt['entity_extract'].format(query=query),
                        'output': think_chain['fine_entity']
                    }
                    entity_lines.append(json.dumps(entity_data))
                    # concept
                    concept_data = {
                        'instruction': '',
                        'input': lightrag_prompt['concept_extract'].format(query=query),
                        'output': think_chain['coarse_concept']
                    }
                    concept_lines.append(json.dumps(concept_data))
                    # 构建RAG, 因为他一定需要这个
                    rag_data = {
                        'instruction': '',
                        'input': lightrag_prompt['combined_prompt'].format(subquery=query,
                                                                      kg_results=think_chain["kg_results"],
                                                                      coarse_results=think_chain['coarse_results']),
                        # 构建决定
                        'output': golden_answer
                    }
                    rag_lines.append(json.dumps(rag_data))
                else:
                    wrong_data = {
                        'instruction': '',
                        'input': query,  # 构建决定
                        'output': golden_answer
                    }
                    wrong_lines.append(json.dumps(wrong_data))

                pure_path_lines.append(json.dumps(pure_path_data))

            # 存储
            with open(os.path.join(to_path, '{}_pure_path.jsonl'.format(config['MODEL'])), 'w') as f:
                f.write('\n'.join(pure_path_lines))

            with open(os.path.join(to_path, '{}_entity.jsonl'.format(config['MODEL'])), 'w') as f:
                f.write('\n'.join(entity_lines))
            with open(os.path.join(to_path, '{}_concept.jsonl'.format(config['MODEL'])), 'w') as f:
                f.write('\n'.join(concept_lines))

            with open(os.path.join(to_path, '{}_rag.jsonl'.format(config['MODEL'])), 'w') as f:
                f.write('\n'.join(rag_lines))

            with open(os.path.join(to_path, '{}_wrong.jsonl'.format(config['MODEL'])), 'w') as f:
                f.write('\n'.join(rag_lines))

            all_data = []
            all_data.extend(entity_lines)
            all_data.extend(concept_lines)
            all_data.extend(rag_lines)
            all_data.extend(wrong_lines)
            with open(os.path.join(to_path, '{}_all_data.jsonl'.format(config['MODEL'])), 'w') as f:  # 包含所有中间步骤
                f.write('\n'.join(all_data))
        elif config['MODEL'] == 'KARE':
            print("KARE model, no need to reconstruct the data, just use the original data.")
            rag_lines = []
            wrong_lines = []  # LLM warm start没有回答正确直接记录答案

            pure_path_lines = []

            for line in lines:
                try:
                # print(line)
                    line = eval(line)
                except:
                    continue
                think_chain, query, golden_answer, final_answer, decision = line['think_chain'], line['query'], line[
                    'groundtruth'], line['final_answer'], line['decision']  # 这里不是那么简单似乎

                pure_path_data = {
                    'instruction': '',
                    'input': query,  # 构建决定
                    'output': golden_answer
                }

                if final_answer == 'True':
                    # 构建decision
                    # 构建RAG
                    rag_data = {
                        'instruction': '',
                        'input': kare_prompt['combined_prompt'].format(subquery=query,
                                                                      kg_results=think_chain["kg_results"],
                                                                      com_results=think_chain['com_results']),
                        # 构建决定
                        'output': golden_answer + '.\n Reason: ' + think_chain['reason'] # 需要把思维接上
                    }
                    rag_lines.append(json.dumps(rag_data))
                else:
                    wrong_data = {
                        'instruction': '',
                        'input': query,  # 构建决定
                        'output': golden_answer
                    }
                    wrong_lines.append(json.dumps(wrong_data))

                pure_path_lines.append(json.dumps(pure_path_data))

            # 存储
            with open(os.path.join(to_path, '{}_pure_path.jsonl'.format(config['MODEL'])), 'w') as f:
                f.write('\n'.join(pure_path_lines))

            with open(os.path.join(to_path, '{}_rag.jsonl'.format(config['MODEL'])), 'w') as f:
                f.write('\n'.join(rag_lines))

            with open(os.path.join(to_path, '{}_wrong.jsonl'.format(config['MODEL'])), 'w') as f:
                f.write('\n'.join(rag_lines))

            all_data = []
            all_data.extend(rag_lines)
            all_data.extend(wrong_lines)
            with open(os.path.join(to_path, '{}_all_data.jsonl'.format(config['MODEL'])), 'w') as f:  # 包含所有中间步骤
                f.write('\n'.join(all_data))
        elif config['MODEL'] == 'MedRAG':
            print("KARE model, no need to reconstruct the data, just use the original data.")
            rag_lines = []
            wrong_lines = []  # LLM warm start没有回答正确直接记录答案

            pure_path_lines = []

            for line in lines:
                print(line)
                try: # 额外异常
                    line = eval(line)
                except:
                    continue
                think_chain, query, golden_answer, final_answer, decision = line['think_chain'], line['query'], line[
                    'groundtruth'], line['final_answer'], line['decision']  # 这里不是那么简单似乎

                pure_path_data = {
                    'instruction': '',
                    'input': query,  # 构建决定
                    'output': golden_answer
                }

                if final_answer == 'True': # 他也是一定要RAG
                    # 构建decision
                    # 构建RAG
                    rag_data = {
                        'instruction': '',
                        'input': medrag_prompt['combined_prompt'].format(subquery=query,
                                                                      kg_results=think_chain["kg_results"],
                                                                      disease_results=think_chain['disease_results']),
                        # 构建决定
                        'output': golden_answer
                    }
                    rag_lines.append(json.dumps(rag_data))
                else:
                    wrong_data = {
                        'instruction': '',
                        'input': query,  # 构建决定
                        'output': golden_answer
                    }
                    wrong_lines.append(json.dumps(wrong_data))

                pure_path_lines.append(json.dumps(pure_path_data))

            # 存储
            with open(os.path.join(to_path, '{}_pure_path.jsonl'.format(config['MODEL'])), 'w') as f:
                f.write('\n'.join(pure_path_lines))

            with open(os.path.join(to_path, '{}_rag.jsonl'.format(config['MODEL'])), 'w') as f:
                f.write('\n'.join(rag_lines))

            with open(os.path.join(to_path, '{}_wrong.jsonl'.format(config['MODEL'])), 'w') as f:
                f.write('\n'.join(rag_lines))

            all_data = []
            all_data.extend(rag_lines)
            all_data.extend(wrong_lines)
            with open(os.path.join(to_path, '{}_all_data.jsonl'.format(config['MODEL'])), 'w') as f:  # 包含所有中间步骤
                f.write('\n'.join(all_data))
    else:
        pure_path_lines = []
        # test不需要构建，因为完全没见过，只要inference
        for line in lines:
            line = eval(line)
            query, golden_answer = line['query'], line['groundtruth']

            pure_path_data = {
                'instruction': '',
                'input': query,  # 构建决定
                'output': golden_answer
            }
            pure_path_lines.append(json.dumps(pure_path_data))
        with open(os.path.join(to_path, '{}_pure_path_test.jsonl'.format(config['MODEL'])), 'w') as f:
            f.write('\n'.join(pure_path_lines))

    print("reconstruct Done!")




def create_sft_data_infer_flask(dataloader, output_path, config, map_dic, task_mode='ehr', train_mode=False,run_config:RunnableConfig=RunnableConfig(llm={}), topk=1):
    # 处理最简单
    # 除了COT过程外都一摸一样
    # 1. 加载LLM模型和分词器
    if not train_mode:
        with open(output_path, 'w') as json_file:  # 使用 'w' 模式写入数据
            total_batches = len(dataloader)  # 获取总批次数量（若dataloader支持）
            for index, batch in enumerate(tqdm(dataloader, desc="Processing batches", total=total_batches)):
                # if index > 3:  # 仅使用部分的数据。调模型
                #     break
            # for index, batch in enumerate(dataloader):
                if task_mode == 'ehr':  # EHR task任务
                    batch = batch_encode(batch, feature_keys=config['FEATURE'], task=config['TASK'],
                                         max_length=config['MAXSEQ'], max_num=config['MAXCODESEQ'], map_dic=map_dic)
                    # 创建actor实例
                    for _, (condition, procedure, medication, ground_truth) in enumerate(batch):
                        query = task_templates[config['TASK']].format(disease_info=condition, procedure_info=procedure,
                                                                      prescription_info=medication)
                        groundtruth = ground_truth

                        # 假设这是你的字典数据
                        data_dict = {
                            "query": query,
                            "groundtruth": groundtruth,
                        }
                        json.dump(data_dict, json_file)
                        json_file.write('\n')  # 每个批次结果换行

                elif task_mode in ['mqa', 'qa', 'summary']:
                    # 创建actor实例
                    input_querys, targets = batch['input'], batch['output']
                    for _, (input_query, target) in enumerate(zip(input_querys, targets)):
                        query = task_templates[task_mode].format(query=input_query)
                        groundtruth = target

                        # 假设这是你的字典数据
                        data_dict = {
                            "query": query,
                            "groundtruth": groundtruth,
                        }
                        json.dump(data_dict, json_file)
                        json_file.write('\n')  # 每个批次结果换行

                    # if index > 2:  # 仅使用部分的数据。调模型
                    #     break

        print("Test. You have finished in the pipeline!")
        return
    else:
        llm = None
        embedding_model = None

        # Step 1. data
        # 2. 创建异构图 & text chunk， 这些都是task & dataset共享的。
        root_to_dir = "/hpc2hdd/home/xxxs349/xxxc/RAGHealth/data/ready" # 这个路径都是公用的
        if not os.path.exists(os.path.join(root_to_dir, 'hetero_graph.dgl')):
            graph, node_data, meta_paths = create_hetero_graphs(config['KG_PATH'], root_to_dir) # kg_data用于存储
            # 2. 定义元路径, 需要filter掉一些
            filter_list = [('drug', 'drug_effect', 'effect/phenotype'), ('drug', 'drug_drug', 'drug'), ('disease', 'disease_phenotype_positive', 'effect/phenotype'),('gene/protein', 'disease_protein', 'disease')]
            meta_paths = [i for i in meta_paths if i not in filter_list]
            metapaths_dic = create_meta_path(llm, meta_paths, root_to_dir) # kg_data用于存储
        else: # 存储有问题
            node_data, graph, metapaths_dic = load_graph_data(root_to_dir)
            meta_paths = [metapaths_dic[i]['raw-meta-path'] for i in metapaths_dic]

        if not os.path.exists(os.path.join(root_to_dir, 'text_chunk.pkl')): # 功能已经移到rag_flask中
            # 耗时比较大，不建议和上面合并
            text_chunks = create_naive_chunks(config['TEXT_PATH'], root_to_dir)
        else:
            text_chunks = load_text_chunk_data(root_to_dir)

        if not os.path.exists(os.path.join(root_to_dir, 'user_chunk.pkl')) and train_mode:  # 功能已经移到rag_flask中
            # 耗时比较大，不建议和上面合并
            user_chunks = create_user_chunks(dataloader, root_to_dir, task_mode, map_dic) # 只用train_loader
        else:
            user_chunks = load_user_chunk_data(root_to_dir)

        print('Agent Done! No need load graph. But you need to run the flask server!')


        # Step 3. continue the pipeline
        # 3. 运行 Pipeline
        with open(output_path, 'w', buffering=1) as json_file:  # 使用 'w' 模式写入数据
            total_batches = len(dataloader)  # 获取总批次数量（若dataloader支持）
            for index, batch in enumerate(tqdm(dataloader, desc="Processing batches", total=total_batches)):
            # for index, batch in enumerate(dataloader):
                results = []
                if task_mode == 'ehr': # EHR task任务
                    batch = batch_encode(batch, feature_keys=config['FEATURE'], task=config['TASK'], max_length=config['MAXSEQ'], max_num=config['MAXCODESEQ'], map_dic=map_dic)
                    # 创建actor实例
                    for _, (condition, procedure, medication, ground_truth) in enumerate(batch):
                        query = task_templates[config['TASK']].format(disease_info=condition, procedure_info=procedure,
                                                                      prescription_info=medication)
                        ground_truth = ground_truth
                        results.append([query, ground_truth])

                elif task_mode in ['mqa','qa','summary']:
                     # 创建actor实例
                     input_querys, targets = batch['input'], batch['output']
                     for _, (input_query, target) in enumerate(zip(input_querys, targets)):
                         query = task_templates[task_mode].format(query=input_query)
                         ground_truth = target
                         results.append([query, ground_truth])

                # 每个actor进行action
                for batch_data in results:
                    query, groundtruth = batch_data # [{}], [[{}],[{}]], '', ''
                    # 假设这是你的字典数据
                    data_dict = {
                        "query": query,
                        "groundtruth": groundtruth,
                    }
                    json.dump(data_dict, json_file)
                    json_file.write('\n')  # 每个批次结果换行

                # if index > 1: # 仅使用部分的数据。 调模型
                #     break

        print("Train. You have finished in the pipeline!")




def re_construct_format_infer(path, to_path, train_mode=True, config=None):
    """
    将数据转换为 LLM Factory 格式, ours
    """
    with open(path, 'r') as f:
        lines = f.readlines() # all data
    if train_mode:
        print("Non infer model, no need to reconstruct the data, just use the original data.")
        wrong_lines = [] # LLM rollout没有回答正确,直接记录答案.增加数据利用。
        pure_path_lines = []


        for line in lines:
            line = eval(line)
            query, golden_answer = line['query'], line['groundtruth']# 这里不是那么简单似乎

            pure_path_data = {
                'instruction': '',
                'input': query,  # 构建决定
                'output': golden_answer
            }

            wrong_data = {
                'instruction': '',
                'input': query,  # 构建决定
                'output': golden_answer
            }
            wrong_lines.append(json.dumps(wrong_data)) # 这里本质上就是一摸一样了

            pure_path_lines.append(json.dumps(pure_path_data))

        # 存储
        with open(os.path.join(to_path, '{}_pure_path.jsonl'.format(config['MODEL'])), 'w') as f:
            f.write('\n'.join(pure_path_lines))


        all_data = [] # 记录rollout所有中间动作，记录为pair
        all_data.extend(wrong_lines)
        with open(os.path.join(to_path, '{}_all_data.jsonl'.format(config['MODEL'])), 'w') as f: # 包含所有中间步骤
            f.write('\n'.join(all_data))

    else:
        pure_path_lines = []
        # test不需要构建，因为完全没见过，只要inference
        for line in lines:
            line = eval(line)
            query, golden_answer = line['query'], line['groundtruth']

            pure_path_data = {
                'instruction': '',
                'input': query,  # 构建决定
                'output': golden_answer
            }
            pure_path_lines.append(json.dumps(pure_path_data))
        with open(os.path.join(to_path, '{}_pure_path_test.jsonl'.format(config['MODEL'])), 'w') as f:
            f.write('\n'.join(pure_path_lines))

    print("reconstruct Done!")





def create_sft_data_flash_flask(dataloader, output_path, config, map_dic, task_mode='ehr', train_mode=False, run_config:RunnableConfig=RunnableConfig(llm={}), topk=1):
    # 处理最简单
    # 除了COT过程外都一摸一样
    # 1. 加载LLM模型和分词器
    if not train_mode:
        with open(output_path, 'w') as json_file:  # 使用 'w' 模式写入数据
            total_batches = len(dataloader)  # 获取总批次数量（若dataloader支持）
            for index, batch in enumerate(tqdm(dataloader, desc="Processing batches", total=total_batches)):
            # for index, batch in enumerate(dataloader):
                if task_mode == 'ehr':  # EHR task任务
                    batch = batch_encode(batch, feature_keys=config['FEATURE'], task=config['TASK'],
                                         max_length=config['MAXSEQ'], max_num=config['MAXCODESEQ'], map_dic=map_dic)
                    # 创建actor实例
                    for _, (condition, procedure, medication, ground_truth) in enumerate(batch):
                        query = task_templates[config['TASK']].format(disease_info=condition, procedure_info=procedure,
                                                                      prescription_info=medication)
                        groundtruth = ground_truth

                        # 假设这是你的字典数据
                        data_dict = {
                            "query": query,
                            "groundtruth": groundtruth,
                        }
                        json.dump(data_dict, json_file)
                        json_file.write('\n')  # 每个批次结果换行

                        # if index > 1:  # 仅使用部分的数据。
                        #     break

                elif task_mode in ['mqa', 'qa', 'summary']:
                    # 创建actor实例
                    input_querys, targets = batch['input'], batch['output']
                    for _, (input_query, target) in enumerate(zip(input_querys, targets)):
                        query = task_templates[task_mode].format(query=input_query)
                        groundtruth = target

                    # 假设这是你的字典数据
                    data_dict = {
                        "query": query,
                        "groundtruth": groundtruth,
                    }
                    json.dump(data_dict, json_file)
                    json_file.write('\n')  # 每个批次结果换行

                    # if index > 1:  # 仅使用部分的数据。
                    #     break

        print("Test. You have finished in the pipeline!")
        return
    else:
        llm = None
        embedding_model = None

        # Step 1. data
        # 2. 创建异构图 & text chunk， 这些都是task & dataset共享的。
        root_to_dir = "/hpc2hdd/home/xxxs349/xxxc/RAGHealth/data/ready" # 这个路径都是公用的
        if not os.path.exists(os.path.join(root_to_dir, 'hetero_graph.dgl')):
            graph, node_data, meta_paths = create_hetero_graphs(config['KG_PATH'], root_to_dir) # kg_data用于存储
            # 2. 定义元路径, 需要filter掉一些
            filter_list = [('drug', 'drug_effect', 'effect/phenotype'), ('drug', 'drug_drug', 'drug'), ('disease', 'disease_phenotype_positive', 'effect/phenotype'),('gene/protein', 'disease_protein', 'disease')]
            meta_paths = [i for i in meta_paths if i not in filter_list]
            metapaths_dic = create_meta_path(llm, meta_paths, root_to_dir) # kg_data用于存储
        else: # 存储有问题
            node_data, graph, metapaths_dic = load_graph_data(root_to_dir)
            meta_paths = [metapaths_dic[i]['raw-meta-path'] for i in metapaths_dic]

        if not os.path.exists(os.path.join(root_to_dir, 'text_chunk.pkl')): # 功能已经移到rag_flask中
            # 耗时比较大，不建议和上面合并
            text_chunks = create_naive_chunks(config['TEXT_PATH'], root_to_dir)
        else:
            text_chunks = load_text_chunk_data(root_to_dir)

        if not os.path.exists(os.path.join(root_to_dir, 'user_chunk.pkl')) and train_mode:  # 功能已经移到rag_flask中
            # 耗时比较大，不建议和上面合并
            user_chunks = create_user_chunks(dataloader, root_to_dir, task_mode, map_dic) # 只用train_loader
        else:
            user_chunks = load_user_chunk_data(root_to_dir)

        print('Agent Done! No need load graph. But you need to run the flask server!')


        # Step 3. continue the pipeline
        # 3. 运行 Pipeline
        with open(output_path, 'w', buffering=1) as json_file:  # 使用 'w' 模式写入数据
            total_batches = len(dataloader)  # 获取总批次数量（若dataloader支持）
            for index, batch in enumerate(tqdm(dataloader, desc="Processing batches", total=total_batches)):
            # for index, batch in enumerate(dataloader):
                results = []
                if task_mode == 'ehr': # EHR task任务
                    batch = batch_encode(batch, feature_keys=config['FEATURE'], task=config['TASK'], max_length=config['MAXSEQ'], max_num=config['MAXCODESEQ'], map_dic=map_dic)
                    # 创建actor实例
                    for _, (condition, procedure, medication, ground_truth) in enumerate(batch):
                        query = task_templates[config['TASK']].format(disease_info=condition, procedure_info=procedure,
                                                                      prescription_info=medication)
                        ground_truth = ground_truth
                        results.append([query, ground_truth])

                elif task_mode in ['mqa','qa','summary']:
                     # 创建actor实例
                     input_querys, targets = batch['input'], batch['output']
                     for _, (input_query, target) in enumerate(zip(input_querys, targets)):
                         query = task_templates[task_mode].format(query=input_query)
                         ground_truth = target
                         results.append([query, ground_truth])

                # 每个actor进行action
                for batch_data in results:
                    query, groundtruth = batch_data # [{}], [[{}],[{}]], '', ''
                    # 假设这是你的字典数据
                    data_dict = {
                        "query": query,
                        "groundtruth": groundtruth,
                    }
                    json.dump(data_dict, json_file)
                    json_file.write('\n')  # 每个批次结果换行

                # if index > 1: # 仅使用部分的数据。
                #     break

        print("Train. You have finished in the pipeline!")




def re_construct_format_flash(path, to_path, train_mode=True,config=None):
    """
    将数据转换为 LLM Factory 格式, ours
    """
    with open(path, 'r') as f:
        lines = f.readlines() # all data
    if train_mode:
        print("Non infer model, no need to reconstruct the data, just use the original data.")
        wrong_lines = [] # LLM warm start没有回答正确直接记录答案
        pure_path_lines = []


        for line in lines:
            line = eval(line)
            query, golden_answer = line['query'], line['groundtruth']# 这里不是那么简单似乎

            pure_path_data = {
                'instruction': '',
                'input': query,  # 构建决定
                'output': golden_answer
            }

            wrong_data = {
                'instruction': '',
                'input': query,  # 构建决定
                'output': golden_answer
            }
            wrong_lines.append(json.dumps(wrong_data)) # 这里本质上就是一摸一样了

            pure_path_lines.append(json.dumps(pure_path_data))

        # 存储
        with open(os.path.join(to_path, '{}_pure_path.jsonl'.format(config['MODEL'])), 'w') as f:
            f.write('\n'.join(pure_path_lines))


        all_data = []
        all_data.extend(wrong_lines)
        with open(os.path.join(to_path, '{}_all_data.jsonl'.format(config['MODEL'])), 'w') as f: # 包含所有中间步骤
            f.write('\n'.join(all_data))

    else:
        pure_path_lines = []
        # test不需要构建，因为完全没见过，只要inference
        for line in lines:
            line = eval(line)
            query, golden_answer = line['query'], line['groundtruth']

            pure_path_data = {
                'instruction': '',
                'input': query,  # 构建决定
                'output': golden_answer
            }
            pure_path_lines.append(json.dumps(pure_path_data))
        with open(os.path.join(to_path, '{}_pure_path_test.jsonl'.format(config['MODEL'])), 'w') as f:
            f.write('\n'.join(pure_path_lines))

    print("reconstruct Done!")





def create_user_chunks(dataloader, root_to_dir, task_mode, map_dic):
    results = []
    total_batches = len(dataloader)  # 获取总批次数量（若dataloader支持）
    for index, batch in enumerate(tqdm(dataloader, desc="Processing batches", total=total_batches)):
    # for index, batch in enumerate(dataloader):
        if task_mode == 'ehr':  # EHR task任务
            batch = batch_encode(batch, feature_keys=config['FEATURE'], task=config['TASK'],
                                 max_length=config['MAXSEQ'], max_num=config['MAXCODESEQ'], map_dic=map_dic)
            for _, (condition, procedure, medication, ground_truth) in enumerate(batch):
                results.append(str(condition)+str(procedure)+str(medication) + " Answer: " + ground_truth)
        elif task_mode in ['mqa', 'qa', 'summary']:
            # 创建actor实例
            input_querys, targets = batch['input'], batch['output']
            for _, (input_query, target) in enumerate(zip(input_querys, targets)):
                input_querys, targets = batch['input'], batch['output']
                results.append(input_query + " Answer: " + target)
        save_pickle(results, os.path.join(root_to_dir, 'user_chunk.pkl'))
        print("Useer chunk have been saved!")
        return results


def create_sft_data_flask(dataloader, output_path, config, map_dic, task_mode='ehr', train_mode=False, run_config:RunnableConfig=RunnableConfig(llm={}), topk=1):
    if not train_mode:
        with open(output_path, 'w') as json_file:  # 使用 'w' 模式写入数据
            total_batches = len(dataloader)  # 获取总批次数量（若dataloader支持）
            for index, batch in enumerate(tqdm(dataloader, desc="Processing batches", total=total_batches)):
            # for index, batch in enumerate(dataloader):
                if task_mode=='ehr':  # EHR task任务
                    batch = batch_encode(batch, feature_keys=config['FEATURE'], task=config['TASK'],
                                         max_length=config['MAXSEQ'], max_num=config['MAXCODESEQ'], map_dic=map_dic)
                    # 创建actor实例
                    for _, (condition, procedure, medication, ground_truth) in enumerate(batch):
                        query = task_templates[config['TASK']].format(disease_info=condition, procedure_info=procedure,
                                                                     prescription_info=medication)
                        groundtruth = ground_truth

                        # 假设这是你的字典数据
                        data_dict = {
                            "golden_paths": [],  # 保证格式一致
                            "negative_paths": [],
                            "query": query,
                            "groundtruth": groundtruth
                        }
                        json.dump(data_dict, json_file)
                        json_file.write('\n')  # 每个批次结果换行

                        # if index > 1:  # 仅使用部分的数据。
                        #     break

                elif task_mode in ['mqa', 'qa', 'summary']:
                    # 创建actor实例
                    input_querys, targets = batch['input'], batch['output']
                    for _, (input_query, target) in enumerate(zip(input_querys, targets)):
                        query=task_templates[task_mode].format(query=input_query)
                        groundtruth = target

                        # 假设这是你的字典数据
                        data_dict = {
                            "golden_paths": [],  # 保证格式一致
                            "negative_paths": [],
                            "query": query,
                            "groundtruth": groundtruth
                        }
                        json.dump(data_dict, json_file)
                        json_file.write('\n')  # 每个批次结果换行

                    # if index > 1:  # 仅使用部分的数据。
                    #     break

        print("Test. You have finished in the pipeline!")
        return
    else:
        # 注意format 适用于summary任务，本身传进来就是正常的QA
        # 这里可以使用flask进行分布式
        # 1. 加载LLM模型和分词器, 这里估计要重新创建了
        llm = None#get_llm_model(config, config['T_LLM'], api_base=None, path=config['T_LLM_PATH'])
        embedding_model = None#get_emb_model(config)

        # print("AAAAAAA", llm.model.device, embedding_model.device)

        # 2. 创建异构图 & text chunk， 这些都是task & dataset共享的。
        root_to_dir = "/hpc2hdd/home/xxxs349/xxxc/RAGHealth/data/ready" # 这个路径都是公用的
        if not os.path.exists(os.path.join(root_to_dir, 'hetero_graph.dgl')):
            graph, node_data, meta_paths = create_hetero_graphs(config['KG_PATH'], root_to_dir) # kg_data用于存储
            # 2. 定义元路径, 需要filter掉一些, 这些太大了，存不起
            filter_list = [('drug', 'drug_effect', 'effect/phenotype'), ('drug', 'drug_drug', 'drug'), ('disease', 'disease_phenotype_positive', 'effect/phenotype'),('gene/protein', 'disease_protein', 'disease')]
            meta_paths = [i for i in meta_paths if i not in filter_list]
            metapaths_dic = create_meta_path(llm, meta_paths, root_to_dir) # kg_data用于存储
        else: # 存储有问题
            node_data, graph, metapaths_dic = load_graph_data(root_to_dir)
            meta_paths = [metapaths_dic[i]['raw-meta-path'] for i in metapaths_dic]

        if not os.path.exists(os.path.join(root_to_dir, 'text_chunk.pkl')): # 功能已经移到rag_flask中
            # 耗时比较大，不建议和上面合并
            text_chunks = create_naive_chunks(config['TEXT_PATH'], root_to_dir)
        else:
            text_chunks = load_text_chunk_data(root_to_dir)

        if not os.path.exists(os.path.join(root_to_dir, 'user_chunk.pkl')) and train_mode:  # 功能已经移到rag_flask中
            # 耗时比较大，不建议和上面合并
            user_chunks = create_user_chunks(dataloader, root_to_dir, task_mode, map_dic) # 只用train_loader
        else:
            user_chunks = load_user_chunk_data(root_to_dir)

        metapaths_dic = {key: value for key, value in metapaths_dic.items() if key in map(str, range(0, 10))}
        print('Agent Done! No need load graph. But you need to run the flask server!')
        # 3. 运行 Pipeline
        gpu_count = torch.cuda.device_count()

        ray.init(num_gpus=gpu_count) # 用4个gpu，供所有actor可见， 最好和batch相同或者可以被batch整除
        # # 创建类的实例, 不传入gpus则默认在CPU上面运行，
        ray_workers =  [AgentPipeline.remote(llm, embedding_model, metapaths_dic, max_iter=config['DEPTH'], ratio=config['META_RATIO'], topk=config['TOPK'], config=config) for i in range(gpu_count)]

        # 创建 4 个 Actor 实例，分布在 4 个 GPU 上。俺不知道为啥为空。
        print("RAY USE GPU IDS", ray.get_gpu_ids())

        # 调用类方法
        with open(output_path, 'w', buffering=2048) as json_file:  # 使用 'w' 模式写入数据
            # json_file.write = lambda data: json_file.write(data) or json_file.flush() # 不知道会不会变快

            total_batches = len(dataloader)  # 获取总批次数量（若dataloader支持）
            for index, batch in enumerate(tqdm(dataloader, desc="Processing batches", total=total_batches)):
                if task_mode == 'ehr': # EHR task任务
                    batch = batch_encode(batch, feature_keys=config['FEATURE'], task=config['TASK'], max_length=config['MAXSEQ'], max_num=config['MAXCODESEQ'], map_dic=map_dic)
                    # 创建actor实例
                    # futures = [ray_workers[inde].run.remote(
                    #     query=task_templates[config['TASK']].format(disease_info=condition, procedure_info=procedure,
                    #                                                 prescription_info=medication),
                    #     ground_truth=ground_truth)
                    #            for inde, (condition, procedure, medication, ground_truth) in enumerate(batch)]
                    futures = []
                    batch = list(batch)
                    mini_batch_size = config['BATCH'] // gpu_count
                    for i in range(0, len(batch), mini_batch_size):
                        # 提取mini-batch并准备输入
                        mini_batch = batch[i:i + mini_batch_size]
                        inputs = [
                            (task_templates[config['TASK']].format(
                                disease_info=cond,
                                procedure_info=proc,
                                prescription_info=med
                            ), ground_truth)
                            for cond, proc, med, ground_truth in mini_batch
                        ]
                        inputs, ground_truths = zip(*inputs)


                        # 分配给worker（循环使用）
                        worker_idx = i // mini_batch_size % len(ray_workers)
                        futures.append(ray_workers[worker_idx].batch_run.remote(inputs, ground_truths))
                elif task_mode in ['mqa','qa','summary']:
                     # 创建actor实例
                     # input_querys, targets = batch['input'], batch['output']
                     # futures = [ray_workers[inde].run.remote(
                     #     query=task_templates[task_mode].format(query=input_query),
                     #     ground_truth=target)
                     #     for inde, (input_query, target) in enumerate(zip(input_querys, targets))]
                    futures = []
                    input_querys, targets = batch['input'], batch['output']
                    mini_batch_size = config['BATCH'] // gpu_count
                    for i in range(0, len(input_querys), mini_batch_size):
                        # 提取mini-batch并准备输入
                        mini_batch_querys = input_querys[i:i + mini_batch_size]
                        mini_batch_targets = targets[i:i + mini_batch_size]
                        mini_batch_querys = [task_templates[task_mode].format(query=input_query) for input_query in
                                             mini_batch_querys]

                        # 分配给worker（循环使用）
                        worker_idx = i // mini_batch_size % len(ray_workers)
                        futures.append(ray_workers[worker_idx].batch_run.remote(mini_batch_querys, mini_batch_targets,
                                                                                run_config=run_config, topk=topk))

                # 每个actor进行action
                results =  get_results_parallel(futures) #ray.get(futures)

                for mini_batch_data in results:
                    for batch_data in mini_batch_data:
                        golden_paths, negative_paths, query, groundtruth = batch_data # [{}], [[{}],[{}]], '', ''
                        # 假设这是你的字典数据
                        data_dict = {
                            "golden_paths": golden_paths, # 如果golden_path为空，则证明在setting内没有找到答案。
                            "negative_paths": negative_paths,
                            "query": query,
                            "groundtruth": groundtruth
                        }
                        json.dump(data_dict, json_file)
                        json_file.write('\n')  # 每个批次结果换行

                # if index > 1: # 仅使用部分的数据。
                #     break

        print("Train. You have finished in the pipeline!")



def re_construct_format(path, to_path, metapaths, train_mode=True):
    """
    将数据转换为 LLM Factory 格式, ours
    """
    # 读取meta_path, 公用
    meta_descriptions = "\n".join([
        f"ID: {index}\nMeta_path: {meta['meta-path']}\nDescription: {meta['description']}\n"
        for index, meta in metapaths.items()
    ]) # 和agent_top中 meta_description保持一致

    with open(path, 'r') as f:
        lines = f.readlines() # all data
    if train_mode:
        rewrite_lines = [] # rewrite阶段
        decision_lines = []
        sub_response_lines = []
        meta_chose_lines = []
        follow_lines = []
        terminal_lines = []
        final_answer_lines = []


        no_gold_path_lines = [] # warm-start阶段没有的QA。
        pure_path_lines = []


        double_label = None
        print(config['TASK'])
        # 用处不大感觉。
        if config['TASK'] in ['MOR', 'IHM']:
            double_label = 'yes'
            if config['DATASET'] == 'MIII':
                double_num = 4
            elif config['DATASET'] == 'eICU':
                double_num = 5
            elif config['DATASET']=='MIV':
                double_num = 2
        elif config['TASK'] in ['REA']:
            double_label = 'no'
            if config['DATASET'] == 'eICU': # 只有你不均衡在这个任务上。
                double_num = 5
            elif config['DATASET'] == 'MIII': 
                double_num = 2
            elif config['DATASET'] == 'MIV':
                double_num = 1   # REA都不能翻倍
        elif config['TASK'] in ['LOS']:
            double_label = 'multi'
            double_num = 1
        print("Current double_label", double_label)
                

        for line in lines:
            line = eval(line)
            gold_path, query, golden_answer = line['golden_paths'], line['query'], line['groundtruth']
            reason_len = len(gold_path)

            pure_path_data = {
                'instruction': '',
                'input': query,  # 构建决定
                'output': golden_answer
            }
            pure_path_lines.append(json.dumps(pure_path_data))

            no_gold_path_data = {
                'instruction': '',
                'input': prompt_templates['complete_checking'].format(query=query, reason_history=''),  # 本质就是记住那些无法回答的，让LLM第一次就回答对
                'output': golden_answer
            }

            if double_label is None:
                no_gold_path_lines.append(json.dumps(no_gold_path_data))

                # print("No need Double.")
            elif double_label in ['multi']:
                # no_gold_path_lines.append(json.dumps(no_gold_path_data))
                if config['DATASET'] in ['MIV', 'eICU']:
                    no_gold_path_lines.append(json.dumps(no_gold_path_data))
                elif config['DATASET'] in ['MIII']:
                    temp_input = no_gold_path_data.copy()
                    temp_input['input'] = get_dropped_text(temp_input['input'],datas=config['DATASET']) # new input
                    no_gold_path_lines.append(json.dumps(temp_input))
            elif double_label in ['yes', 'no']:
                if golden_answer == double_label: # yes no 任务； REA为no，其他为True。
                    # no_gold_path_lines.append(json.dumps(no_gold_path_data))
                    for i in range(double_num):
                        # temp_input = no_gold_path_data.copy()
                        # temp_input['input'] = get_dropped_text(temp_input['input'],datas=config['DATASET']) # new input
                        # no_gold_path_lines.append(json.dumps(temp_input)) # 正例embedding加强。
                        no_gold_path_lines.append(json.dumps(no_gold_path_data))
                else:
                    # temp_input = no_gold_path_data.copy()
                    # temp_input['input'] = get_dropped_text(temp_input['input'],datas=config['DATASET']) # new input

                    no_gold_path_lines.append(json.dumps(no_gold_path_data)) # 只对no进行
                    
            elif double_label == 'multi':
                temp_input = no_gold_path_data.copy()
                temp_input['input'] = get_dropped_text(temp_input['input'],datas=config['DATASET'])
                no_gold_path_lines.append(json.dumps(temp_input))
                # no_gold_path_lines.append(json.dumps(no_gold_path_data)) # 只对no进行
            
            # no_gold_path_lines.append(json.dumps(no_gold_path_data))
                # continue # 仅使用那些有golden path的
            if reason_len == 0: # 不进行下列的。
                continue
            else:
                gold_path = gold_path[0]
            for index, path in enumerate(gold_path): # 一个patient的
                # path = path[0] # [{}]->字典, 因为agent_top, [].append([{}])用了append [{}]。这里务必确定是对的。
                source, subquery, subanswer, decision = path['source'], path['subquery'], path['subanswer'], path['decision']
                chosen_metapaths, kg_results, naive_results = path['chosen_metapaths'], path['chosen_entities'], path['naive_results']
                reason = gold_path[index-1]['reason_history'] if index >0 else ''
                cur_reason, follow = path['reason_history'], path['follow']

                if index ==0: # 第一次要把rewrite加进去
                    rewrite = path['rewrite']
                    rewrite_data = {
                        'instruction': '',
                        'input': prompt_templates['rewrite'].format(query=query),  # 构建决定
                        'output': rewrite
                    }
                    rewrite_lines.append(json.dumps(rewrite_data)) # 这里是直接rewrite的

                # decision
                decision_data = {
                    'instruction': '',
                    'input': prompt_templates['decide'].format(subquery=subquery, reason_history=reason), # 构建决定RAG还是LLM.
                    'output': decision
                }
                decision_lines.append(json.dumps(decision_data))
                # meta_path & sub answer
                if source == 'LLM':
                    sub_response_data = {
                        'instruction': '',
                        'input': prompt_templates['direct_answer'].format(subquery=subquery, reason_history=reason), # 构建决定
                        'output': subanswer
                    }
                    sub_response_lines.append(json.dumps(sub_response_data))
                else:
                    meta_chose_data = {
                        'instruction': '',
                        'input': prompt_templates['meta_path'].format(subquery=subquery, reason_history=reason, meta_path=meta_descriptions), # 构建决定
                        'output': str(chosen_metapaths) # 不能存列表
                    }
                    meta_chose_lines.append(json.dumps(meta_chose_data))

                    sub_response_data = {
                        'instruction': '',
                        'input': prompt_templates['combined_prompt'].format(subquery=subquery, kg_results=kg_results, naive_results=naive_results), # 构建决定
                        'output': subanswer
                    }
                    sub_response_lines.append(json.dumps(sub_response_data))
                # terminal
                if index < len(gold_path)-1: # 模型好像不会说incomplete. 这里不对，这里写的不对，应该是len(). 21个，很少
                    final_answer = {
                        'instruction': '',
                        'input': prompt_templates['complete_checking'].format(query=query, reason_history=cur_reason), # 构建决定
                        'output': 'incomplete'
                    }
                    final_answer_lines.append(json.dumps(final_answer)) # whether deep think
                else:
                    terminal_data = { # 有golden path都是有答案的
                        'instruction': '',
                        'input': prompt_templates['complete_checking'].format(query=query, reason_history=cur_reason), # 构建决定, 会加强原来的，不太好
                        'output': golden_answer
                    }
                    terminal_lines.append(json.dumps(terminal_data))

                # follow up
                if len(gold_path)>0 and index <  len(gold_path) - 1: # 只要不是唯一或者最后一个，就有follow,
                    follow_data = {
                        'instruction': '',
                        'input': prompt_templates['follow'].format(query=query, reason_history=cur_reason), # 构建决定
                        'output': follow
                    }
                    follow_lines.append(json.dumps(follow_data))

        # 存储
        with open(os.path.join(to_path, 'rewrite.jsonl'), 'w') as f:
            f.write('\n'.join(rewrite_lines))

        with open(os.path.join(to_path, 'decision.jsonl'), 'w') as f:
            f.write('\n'.join(decision_lines))

        with open(os.path.join(to_path, 'meta_chose.jsonl'), 'w') as f:
            f.write('\n'.join(meta_chose_lines))

        with open(os.path.join(to_path, 'sub_response.jsonl'), 'w') as f:
            f.write('\n'.join(sub_response_lines))

        with open(os.path.join(to_path, 'final_answer.jsonl'), 'w') as f:
            f.write('\n'.join(final_answer_lines))

        with open(os.path.join(to_path, 'terminal.jsonl'), 'w') as f:
            f.write('\n'.join(terminal_lines))

        with open(os.path.join(to_path, 'followup.jsonl'), 'w') as f:
            f.write('\n'.join(follow_lines))

        with open(os.path.join(to_path, 'no_gold_path.jsonl'), 'w') as f: # 补足没有golden_path的
            f.write('\n'.join(no_gold_path_lines))

         # 要均衡
        if double_label is None:
            with open(os.path.join(to_path, 'pure_path.jsonl'), 'w') as f:
                f.write('\n'.join(pure_path_lines))
        elif double_label in ['yes', 'no']:
            new_pure = []
            pos_pure, neg_pure = [],[]
            for line in pure_path_lines:
                eval_line = eval(line)
                if eval_line['output'] == double_label:
                    pos_pure.append(line)
                else:
                    neg_pure.append(line)
            need_len = min([len(pos_pure), len(neg_pure)])
            print("Double label ", double_label, need_len)
            new_pure.extend(pos_pure[:need_len]) # 看看要不要扩增10倍，不然会有问题。均衡
            new_pure.extend(neg_pure[:need_len*2]) # 另一个扩增MIV-MOR
            with open(os.path.join(to_path, 'pure_path.jsonl'), 'w') as f:
                f.write('\n'.join(new_pure))
        elif double_label == 'multi':
            new_pure = []
            pure_labels = {str(i)+'days': [] for i in range(10)}
            for line in pure_path_lines:
                eval_line = eval(line)
                output = eval_line['output']
                pure_labels[output].append(line)
            # 获取所有列表的长度
            label_lengths = {label: len(qa_list) for label, qa_list in pure_labels.items()}
            # 找到最小长度L
            min_length = min(label_lengths.values())
            min_length = min(min_length, 100) # 不然PPO阶段贼浪费时间
            print("mini length", min_length)
            for label, qa_list in pure_labels.items():
                new_pure.extend(qa_list[:min_length])
            with open(os.path.join(to_path, 'pure_path.jsonl'), 'w') as f:
                f.write('\n'.join(new_pure))            
            
            
        all_data = []
        if config['DATASET'] + '-'+ config['TASK'] in ['MIII-LOS']:
            format_num = (len(terminal_lines) + len(no_gold_path_lines)) // 200
        elif config['DATASET'] + '-'+ config['TASK'] in ['MIV-LOS']:
            format_num = (len(terminal_lines) + len(no_gold_path_lines)) // 500
        else:
            format_num = (len(terminal_lines) + len(no_gold_path_lines)) // 100


        print('Format num, ', format_num)
    
        ## EHR-MIII-MOR (这种任务数据超级多)
        if config['TASK'] not in ['SUMMARY']:
            all_data.extend(rewrite_lines[:format_num])
            all_data.extend(follow_lines[:format_num])
            if config['DATASET'] + '-'+ config['TASK'] not in ['MIV-MOR', 'eICU-LOS']:
                all_data.extend(sub_response_lines[:format_num]) # MIV-MOR去掉。eICU-LOS也去掉
            if config['DATASET'] + '-'+ config['TASK'] not in ['eICU-LOS']:
                print("KKKKKKKKKK")
                all_data.extend(meta_chose_lines[:format_num]) # eICU-LOS基本要去掉


        all_data.extend(final_answer_lines[:format_num]) # 这玩意必须要有,LOS没用
        
        if double_label not in ['yes', 'no']:
            all_data.extend(decision_lines[:format_num]) 
   
        # yes no任务
        if double_label is None or double_label=='multi':
            all_data.extend(terminal_lines) # 难道terminal有问题？加进来干扰学习？
            print("terminal len", len(terminal_lines))
        elif double_label=='multi': # 感觉没啥用。
            new_terminal = []
            terminal_labels = {str(i)+'days': [] for i in range(10)}
            for line in terminal_lines:
                eval_line = eval(line)
                output = eval_line['output']
                terminal_labels[output].append(line)
            # 获取所有列表的长度
            label_lengths = {label: len(qa_list) for label, qa_list in terminal_labels.items()}
            # 对字典按值升序排序，返回包含(键, 值)的元组列表
            sorted_items = sorted(label_lengths.items(), key=lambda x: x[1])
            # 取前5个元素的键
            smallest_5_keys = [item[0] for item in sorted_items[:5]]
            # 找到最小长度L
            min_length = min(label_lengths.values())
            print("terminal mini length and smallest keys", min_length, smallest_5_keys)
            for label, qa_list in terminal_labels.items():
                if label in smallest_5_keys:
                    new_terminal.extend(qa_list[:]*double_num)
                else:
                    new_terminal.extend(qa_list)
            all_data.extend(new_terminal)
            print("new terminal len,", len(new_terminal))

            
        else:
            pos_terminal, neg_terminal = [],[]
            for line in terminal_lines:
                eval_line = eval(line)
                if eval_line['output'] == double_label:
                    pos_terminal.append(line)
                else:
                    neg_terminal.append(line)
            need_len = min([len(pos_terminal), len(neg_terminal)])
            if double_num ==1 or config['DATASET'] + '-'+ config['TASK'] in ['MIII-REA', 'MIV-REA']:
                print("Balance special.")
                all_data.extend(terminal_lines[:len(terminal_lines)])
                # all_data.extend(pos_terminal[:need_len])
                # all_data.extend(neg_terminal[:need_len])
            else:
                all_data.extend(pos_terminal[:need_len]*double_num) # 看看要不要扩增10倍，不然会有问题。。
                all_data.extend(neg_terminal[:need_len]) # eICU-REA要颠倒dataset加权，miii-rea似乎不用。
            print("terminal len", need_len)
        # 

        
        all_data.extend(no_gold_path_lines) # 补足没有golden_path的; 卧槽，好像没有shuffle

        all_data = random.sample(all_data, len(all_data)) # 卧槽，SFT不会进行shuffle吗


        with open(os.path.join(to_path, 'all_data.jsonl'), 'w') as f:
            f.write('\n'.join(all_data))
    else:
        pure_path_lines = []
        # test不需要构建，因为完全没见过，只要inference
        for line in lines:
            line = eval(line)
            query, golden_answer = line['query'], line['groundtruth']

            pure_path_data = {
                'instruction': '',
                'input': query,  # 构建决定
                'output': golden_answer
            }
            pure_path_lines.append(json.dumps(pure_path_data))
        with open(os.path.join(to_path, 'pure_path_test.jsonl'), 'w') as f:
            f.write('\n'.join(pure_path_lines))

    print("reconstruct Done!")



class SimpleDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r') as file:
            # self.lines = [json.loads(line.strip()) for line in file.readlines()]
            self.lines = []
            print("XXXXXX", data_path)
            if data_path == '/hpc2hdd/home/xxxs349/xxxc/RAGHealth/data/MOR/MIV/processed/test_sft_data2.jsonl': # 临时测试
                for line in file.readlines():
                    data = json.loads(line.strip())
                    if data['groundtruth']=='yes':
                        self.lines.append(data)
                    else:
                        continue
            else:
                for line in file.readlines():
                    data = json.loads(line.strip())
                    self.lines.append(data)
        print("Test length", len(self.lines))
        # print("AAAAAAA", type(self.lines[0]), self.lines[0], self.lines[0].keys())

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        # print("CCCCC", idx)
        return self.lines[idx]  # 返回字符串或进一步处理成张量











############## task info ##############

def convert_dataset(samples, dataset_name=None, task_name=None, valid=True, all=False):
    """避免繁琐的处理"""
    if valid: # valid仅适用于第一次sample dataset的处理
        return SampleEHRDataset(
                        samples,
                        dataset_name=dataset_name,
                        task_name=task_name,
                    )
    else:
        return SampleEHRDatasetSIMPLE(
                        samples,
                        dataset_name=dataset_name,
                        task_name=task_name,
                        all=all
                    )



class SampleEHRDatasetSIMPLE(SampleBaseDataset):
    def __init__(self, samples: List[str], code_vocs=None, dataset_name="", task_name="", all=False):
        super().__init__(samples, dataset_name, task_name)
        # 重写了init函数避免index，那个index似乎只是用于split。
        self.samples = samples
        if all:
            self.input_info: Dict = self._validate() # 别的不需要valid，大大减少时间


    @property
    def available_keys(self) -> List[str]:
        """Returns a list of available keys for the dataset.

        Returns:
            List of available keys.
        """
        keys = self.samples[0].keys()
        return list(keys)

    def _validate(self) -> Dict:
        """ 1. Check if all samples are of type dict. """
        keys = self.samples[0].keys()

        """
        4. For each key, check if it is either:
            - a single value
            - a single vector
            - a list of codes
            - a list of vectors
            - a list of list of codes
            - a list of list of vectors
        Note that a value is either float, int, or str; a vector is a list of float
        or int; and a code is str.
        """
        # record input information for each key
        input_info = {}
        for key in keys:
            """
            4.1. Check nested list level: all samples should either all be
            - a single value (level=0)
            - a single vector (level=1)
            - a list of codes (level=1)
            - a list of vectors (level=2)
            - a list of list of codes (level=2)
            - a list of list of vectors (level=3)
            """
            levels = set([list_nested_levels(s[key]) for s in self.samples[:5]]) # 只取前5个判断足够

            level = levels.pop()[0]

            # flatten the list
            if level == 0:
                flattened_values = [s[key] for s in self.samples]
            elif level == 1:
                flattened_values = [i for s in self.samples for i in s[key]]
            elif level == 2:
                flattened_values = [j for s in self.samples for i in s[key] for j in i]
            else:
                flattened_values = [
                    k for s in self.samples for i in s[key] for j in i for k in j
                ]

            """
            4.2. Check type: the basic type of each element should be float,
            int, or str.
            """
            types = set([type(v) for v in flattened_values[:5]]) # 只取前5个判断足够
            type_ = types.pop()
            """
            4.3. Combined level and type check.
            """
            if level == 0:
                # a single value
                input_info[key] = {"type": type_, "dim": 0}
            elif level == 1:
                # a single vector or a list of codes
                if type_ in [float, int]:
                    # a single vector
                    lens = set([len(s[key]) for s in self.samples])
                    input_info[key] = {"type": type_, "dim": 1, "len": lens.pop()}
                else:
                    # a list of codes
                    # note that dim is different from level here
                    input_info[key] = {"type": type_, "dim": 2}
            elif level == 2:
                # a list of vectors or a list of list of codes
                if type_ in [float, int]:
                    lens = set([len(i) for s in self.samples for i in s[key]])
                    input_info[key] = {"type": type_, "dim": 2, "len": lens.pop()}
                else:
                    # a list of list of codes
                    # note that dim is different from level here
                    input_info[key] = {"type": type_, "dim": 3}
            else:
                # a list of list of vectors
                lens = set([len(j) for s in self.samples for i in s[key] for j in i])
                input_info[key] = {"type": type_, "dim": 3, "len": lens.pop()}

        return input_info

    def __len__(self):
        return len(self.samples)


def generate_patient_group(samples, path):
    last_visits = get_last_visit_sample(samples).values()
    group_patient = {}
    for record in last_visits:
        patient_id = record['patient_id']
        incomplete = np.array(record['incomplete'], dtype=int)
        max_num = len(incomplete) * 3
        if int(max_num * 1/2) <incomplete.sum()<= int(max_num * 2/3): # 2种以上缺失。
            group_patient[patient_id] = 'G1'
        elif int(max_num * 1/3) <incomplete.sum()<=int(max_num * 1/2):
            group_patient[patient_id] = 'G2'
        elif int(max_num * 1/6) <incomplete.sum()<=int(max_num * 1/3):
            group_patient[patient_id] = 'G3'
        elif incomplete.sum()<=int(max_num * 1/6): # 一种集没有缺失。
            group_patient[patient_id] = 'G4' # 没有缺失基本

    def unique_values(dictionary):
        return set(dictionary.values())
    save_pickle(group_patient, path + 'group_patient.pkl')
    print("group patient id generate done!")
    return




def create_label_for_phenotyping(path):
    """
    Create labels for phenotyping task
    """
    path = path + 'mimic-hcup_ccs_2015_definitions.yaml'
    with open(path) as definitions_file:
        definitions = yaml.load(definitions_file, Loader = yaml.FullLoader)

    code_to_group = {}
    # print(definitions['Tuberculosis']) # examine the structure of the definitions
    for group in definitions:
        codes = definitions[group]['codes']
        for code in codes:
            if code not in code_to_group:
                code_to_group[code] = group # 'V103': 'Tuberculosis'
            else:
                assert code_to_group[code] == group

    id_to_group = sorted(definitions.keys())
    group_to_id = dict((x, i) for (i, x) in enumerate(id_to_group)) # coarse-grained label

    return code_to_group, group_to_id, id_to_group


def get_visit_phenotype(code_to_group, diagnose_visit: List[str]):
    cur_labels = []
    for diagnose in diagnose_visit:
        if diagnose not in code_to_group:
            continue
        group = code_to_group[diagnose]
        cur_labels.append(group)
    cur_labels = list(set(cur_labels))
    return cur_labels



# path = '/home/xxxc/MMHealth/data/'
# code_to_group, group_to_id, id_to_group = create_label_for_phenotyping(path)
#
# # basline别的数据集注释掉if，不然报错。
# note_dict = get_note(config, "/home/xxxc/HyperHealth/data/physionet.org/files/mimic-iv-note/2.2/") if config['DATASET'] =='MIV-Note' else None


def extract_patient_note(patient_id, visit_id):
    zero_tenor = [1.0] * 768  # 这个768可以根据不同的PLM进行调整。
    if (patient_id, visit_id) in note_dict.index:  # 这个需要宠幸跑
        # 访问索引并获取 C 列的值
        return note_dict.loc[(patient_id, visit_id), 'note_emb']
    else:
        # 如果没找到，返回 768 维的零向量
        return zero_tenor


####### Phe experiment
def phe_prediction_miii_fn(patient: Patient):
    """
    处理的是一个patient的数据
    Examples:
        {
            'visit_id': '174162',
            'patient_id': '107',
            'conditions': [['139', '158', '237', '99', '60', '101', '51', '54', '53', '133', '143', '140', '117', '138', '55']],
            "incomplete":  [[False, False, False]]
            'procedures': [['4443', '4513', '3995']],
            'drugs_hist': [['1150', '0005', '6365', '0090', '6155', '0005', '0090', '0000', '6373']],
            'drugs':['1150', '0005', '6365', '0090', '6155', '0005', '0090', '0000', '6373'],
            'conditions_raw': ['139', '158', '237', '99', '60', '101', '51', '54', '53', '133', '143', '140', '117', '138', '55']
            'procedures_raw': ['4443', '4513', '3995'],
            'incomplete_raw': [False, False, False],
            'labels':['X','B','C']
        }
    """
    samples = []
    for i in range(len(patient) - 1):  # visit 次数
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]
        # print(visit)
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        next_conditions = next_visit.get_code_list(table="DIAGNOSES_ICD")
        next_pheno = get_visit_phenotype(code_to_group, next_conditions)

        # ATC 3 level 'A04D'
        drugs = [drug[:config['ATCLEVEL'] + 1] for drug in drugs]


        if len(conditions) * len(procedures) * len(drugs) * len(next_pheno) == 0:
            continue

        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs_hist": drugs,
                "visit_id_hist": visit.visit_id,  # ['1']
                "drugs": drugs,  # used for diffusion
                "conditions_raw": conditions,
                "procedures_raw": procedures,
                "labels": next_pheno,
            }
        )

    # exclude: patients with less than 1 visit
    if len(samples) < 1:  # [{visit 1},{visit 2}], 有1的话，其实本身就至少有2次visit
        return []

    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]]
    samples[0]["visit_id_hist"] = [samples[0]["visit_id_hist"]]

    for i in range(1, len(samples)):  # 第二次，到第N次，一个patient创建一个samples数据，迭代创建
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + [
            samples[i]["drugs_hist"]
        ]
        samples[i]["visit_id_hist"] = samples[i - 1]["visit_id_hist"] + [
            samples[i]["visit_id_hist"]
        ]

    # # remove the target drug from the history，disease prediction不需要
    # for i in range(len(samples)): # 都是最后一位
    #     samples[i]["drugs_hist"][i] = ['<pad>','<pad>','<pad>'] # 去掉target
    #     # 时序对齐的padding
    #     samples[i]["drugs_hist"] = [samples[i]["drugs_hist"][i]] + samples[i]["drugs_hist"][:i]
    return samples


def phe_prediction_miv_note_fn(patient: Patient):
    # 先经过task fn。
    samples = []
    for i in range(len(patient) - 1):  # visit 次数
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]
        # print(visit)
        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        next_conditions = next_visit.get_code_list(table="diagnoses_icd")
        next_pheno = get_visit_phenotype(code_to_group, next_conditions)

        # ATC 3 level 'A04D'
        drugs = [drug[:config['ATCLEVEL'] + 1] for drug in drugs]

        # exclude: visits without condition, procedure, or drug code

        if len(conditions) * len(procedures) * len(drugs) * len(next_pheno) == 0:
            continue

        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs_hist": drugs,
                "visit_id_hist": visit.visit_id,  # ['1']
                "note": extract_patient_note(int(patient.patient_id), int(visit.visit_id)),
                # 暂时不把它当模态, 所以有没有都行； 不然会大幅度减少数据
                "drugs": drugs,  # （used for diffusion） cur
                "conditions_raw": conditions,
                "procedures_raw": procedures,
                "labels": next_pheno,
            }
        )

    # exclude: patients with less than 1 visit
    if len(samples) < 1:  # [{visit 1},{visit 2}], 有1的话，其实本身就至少有2次visit
        return []

    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]]
    samples[0]["visit_id_hist"] = [samples[0]["visit_id_hist"]]
    samples[0]["note"] = [samples[0]["note"]]

    for i in range(1, len(samples)):  # 第二次，到第N次，一个patient创建一个samples数据，迭代创建
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + [
            samples[i]["drugs_hist"]
        ]
        samples[i]["visit_id_hist"] = samples[i - 1]["visit_id_hist"] + [
            samples[i]["visit_id_hist"]
        ]
        samples[i]["note"] = samples[i - 1]["note"] + [
            samples[i]["note"]
        ]

    return samples


def phe_prediction_eicu_fn(patient: Patient):
    samples = []
    for i in range(len(patient) - 1):  # visit 次数
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]
        # print(visit)
        conditions = visit.get_code_list(table="diagnosis")
        procedures = visit.get_code_list(table="physicalExam")
        drugs = visit.get_code_list(table="medication")
        next_conditions = next_visit.get_code_list(table="diagnosis")
        next_pheno = get_visit_phenotype(code_to_group, next_conditions)

        # exclude: visits without condition, procedure, or drug code

        if len(conditions) * len(procedures) * len(drugs) * len(next_pheno) == 0:
            continue

        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs_hist": drugs,
                "visit_id_hist": visit.visit_id,  # ['1']
                "drugs": drugs,  # used for diffusion
                "conditions_raw": conditions,
                "procedures_raw": procedures,
                "labels": next_pheno,
            }
        )

    # exclude: patients with less than 1 visit
    if len(samples) < 1:  # [{visit 1},{visit 2}], 有1的话，其实本身就至少有2次visit
        return []

    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]]
    samples[0]["visit_id_hist"] = [samples[0]["visit_id_hist"]]

    for i in range(1, len(samples)):  # 第二次，到第N次，一个patient创建一个samples数据，迭代创建
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + [
            samples[i]["drugs_hist"]
        ]
        samples[i]["visit_id_hist"] = samples[i - 1]["visit_id_hist"] + [
            samples[i]["visit_id_hist"]
        ]
    return samples


######### los experiment


def categorize_los(days: int):
    """Categorizes length of stay into 10 categories.

    One for ICU stays shorter than a day, seven day-long categories for each day of
    the first week, one for stays of over one week but less than two,
    and one for stays of over two weeks.

    Args:
        days: int, length of stay in days

    Returns:
        category: int, category of length of stay
    """
    # ICU stays shorter than a day
    if days < 1:
        return 0
    # each day of the first week
    elif 1 <= days <= 7:
        return days
    # stays of over one week but less than two
    elif 7 < days <= 14:
        return 8
    # stays of over two weeks
    else:
        return 9


####### disease experiment for HyperHealth
def diag_prediction_ehrshot_fn(patient: Patient):
    samples = []
    for i in range(len(patient) - 1):  # visit 次数
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]
        # print(visit)
        conditions = visit.get_code_list(table="ehrshot")
        procedures = visit.get_code_list(table="ehrshot")
        drugs = visit.get_code_list(table="ehrshot")
        next_conditions = next_visit.get_code_list(table="ehrshot")


        # # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) * len(next_conditions) == 0: # 导致上面的filter根本没用。MMHealth要注释
            continue

        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs_hist": drugs,
                "visit_id_hist": visit.visit_id,  # ['1']
                "drugs": drugs,  # used for diffusion
                "conditions_raw": conditions,
                "procedures_raw": procedures,
                "labels": next_conditions,
            }
        )

    # exclude: patients with less than 1 visit
    if len(samples) < 30:  # "#or len(samples) >200:  # [{visit 1},{visit 2}], 有1的话，其实本身就至少有2次visit
        return []

    samples = samples[-20:]  # 这样的话，每个patient最多30个visit

    samples[0]["conditions"] = [samples[0]["conditions"]]  # 这样visit对应的drug labels不用改;这里最少1个也行
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]]

    samples[0]["visit_id_hist"] = [samples[0]["visit_id_hist"]]

    for i in range(1, len(samples)):  # 第二次，到第N次，一个patient创建一个samples数据，迭代创建
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + [
            samples[i]["drugs_hist"]
        ]
        samples[i]["visit_id_hist"] = samples[i - 1]["visit_id_hist"] + [
            samples[i]["visit_id_hist"]
        ]

    return samples


def diag_prediction_omop_fn(patient: Patient):
    """Processes a single patient for the drug recommendation task.
    Examples:
        {
            'visit_id': '174162',
            'patient_id': '107',
            'conditions': [['139', '158', '237', '99', '60', '101', '51', '54', '53', '133', '143', '140', '117', '138', '55']],
            'procedures': [['4443', '4513', '3995']],
            'drugs_hist': [[]],
            'drugs': ['0000', '0033', '5817', '0057', '0090', '0053', '0', '0012', '6332', '1001', '6155', '1001', '6332', '0033', '5539', '6332', '5967', '0033', '0040', '5967', '5846', '0016', '5846', '5107', '5551', '6808', '5107', '0090', '5107', '5416', '0033', '1150', '0005', '6365', '0090', '6155', '0005', '0090', '0000', '6373'],
        }
    """
    samples = []
    for i in range(len(patient) - 1):  # visit 次数
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]
        # print(visit)
        # conditions = visit.get_code_list(table="condition_occurrence")
        conditions = visit.get_code_list(table="procedure_occurrence")
        procedures = visit.get_code_list(table="procedure_occurrence")
        drugs = visit.get_code_list(table="procedure_occurrence")  # drug_exposure很奇怪，会报错，drug_exposure很少
        next_conditions = next_visit.get_code_list(table="procedure_occurrence")


        # ATC 3 level
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) * len(next_conditions) == 0:
            continue

        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs_hist": drugs,
                "visit_id_hist": visit.visit_id,  # ['1']
                "drugs": drugs,  # used for diffusion
                "conditions_raw": conditions,
                "procedures_raw": procedures,
                "labels": next_conditions,
            }
        )

    # print('sample', samples)
    # exclude: patients with less than 2 visit
    if len(samples) < 1:  # [{},{}]; 这里graphcare动了手脚， 至少有三次
        return []

    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]]  # 这里的drugs_hist本质上就是drug

    samples[0]["visit_id_hist"] = [samples[0]["visit_id_hist"]]

    for i in range(1, len(samples)):  # 第二次，到第N次，一个patient创建一个samples数据
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + [
            samples[i]["drugs_hist"]
        ]
        samples[i]["visit_id_hist"] = samples[i - 1]["visit_id_hist"] + [
            samples[i]["visit_id_hist"]
        ]

    return samples


####### rec experiment for HyperHealth
def rec_prediction_miii_fn(patient: Patient):
    samples = []
    for i in range(len(patient)):  # visit 次数
        visit: Visit = patient[i]
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")

        # ATC 3 level
        drugs = [drug[:config['ATCLEVEL'] + 1] for drug in drugs]

        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:  # 导致上面的过滤根本没用
            continue

        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "visit_id_hist": visit.visit_id,  # ['1']
                "drugs_hist": drugs,
                "conditions_raw": conditions,
                "procedures_raw": procedures,
                "labels": drugs,
            }
        )

    # exclude: patients with less than 2 visit
    if len(samples) < 1:  # [{visit 1},{visit 2}]
        return []

    samples[0]["conditions"] = [samples[0]["conditions"]]  # 这样visit对应的drug labels不用改;这里最少1个也行
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]]
    samples[0]["visit_id_hist"] = [samples[0]["visit_id_hist"]]

    for i in range(1, len(samples)):  # 第二次，到第N次，一个patient创建一个samples数据，迭代创建
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + [
            samples[i]["drugs_hist"]
        ]
        samples[i]["visit_id_hist"] = samples[i - 1]["visit_id_hist"] + [
            samples[i]["visit_id_hist"]
        ]

    # remove the target drug from the history
    for i in range(len(samples)):
        samples[i]["drugs_hist"][i] = ['<pad>', '<pad>', '<pad>']  # 去掉target
        # 时序对齐的padding
        samples[i]["drugs_hist"] = [samples[i]["drugs_hist"][i]] + samples[i]["drugs_hist"][:i]
    return samples


def rec_prediction_miv_fn(patient: Patient):
    samples = []
    for i in range(len(patient)):  # visit 次数
        visit: Visit = patient[i]
        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")

        # ATC 3 level
        drugs = [drug[:config['ATCLEVEL'] + 1] for drug in drugs]

        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:  # 导致上面的过滤根本没用
            continue

        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "visit_id_hist": visit.visit_id,  # ['1']
                "drugs_hist": drugs,
                "conditions_raw": conditions,
                "procedures_raw": procedures,
                "labels": drugs,
            }
        )

    # exclude: patients with less than 2 visit
    if len(samples) < 2:  # [{visit 1},{visit 2}]
        return []

    samples[0]["conditions"] = [samples[0]["conditions"]]  # 这样visit对应的drug labels不用改;这里最少1个也行
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]]
    samples[0]["visit_id_hist"] = [samples[0]["visit_id_hist"]]

    for i in range(1, len(samples)):  # 第二次，到第N次，一个patient创建一个samples数据，迭代创建
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + [
            samples[i]["drugs_hist"]
        ]
        samples[i]["visit_id_hist"] = samples[i - 1]["visit_id_hist"] + [
            samples[i]["visit_id_hist"]
        ]

    # remove the target drug from the history
    for i in range(len(samples)):
        samples[i]["drugs_hist"][i] = ['<pad>', '<pad>', '<pad>']  # 去掉target
        # 时序对齐的padding
        samples[i]["drugs_hist"] = [samples[i]["drugs_hist"][i]] + samples[i]["drugs_hist"][:i]

    return samples


def rec_prediction_miv_note_fn(patient: Patient):
    samples = []
    for i in range(len(patient)):  # visit 次数
        visit: Visit = patient[i]
        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")

        # ATC 3 level
        drugs = [drug[:config['ATCLEVEL'] + 1] for drug in drugs]


        # # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:  # 导致上面的过滤根本没用
            continue

        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "note": extract_patient_note(int(patient.patient_id), int(visit.visit_id)),  # 暂时不把它当模态, 所以有没有都行
                "visit_id_hist": visit.visit_id,  # ['1']
                "drugs_hist": drugs,
                "conditions_raw": conditions,
                "procedures_raw": procedures,
                "labels": drugs,
            }
        )

    # exclude: patients with less than 2 visit
    if len(samples) < 2:  # [{visit 1},{visit 2}]
        return []

    samples[0]["conditions"] = [samples[0]["conditions"]]  # 这样visit对应的drug labels不用改;这里最少1个也行
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]]
    samples[0]["visit_id_hist"] = [samples[0]["visit_id_hist"]]
    samples[0]["note"] = [samples[0]["note"]]

    for i in range(1, len(samples)):  # 第二次，到第N次，一个patient创建一个samples数据，迭代创建
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + [
            samples[i]["drugs_hist"]
        ]
        samples[i]["visit_id_hist"] = samples[i - 1]["visit_id_hist"] + [
            samples[i]["visit_id_hist"]
        ]
        samples[i]["note"] = samples[i - 1]["note"] + [
            samples[i]["note"]
        ]

    # remove the target drug from the history
    for i in range(len(samples)):
        samples[i]["drugs_hist"][i] = ['<pad>', '<pad>', '<pad>']  # 去掉target
        # 时序对齐的padding
        samples[i]["drugs_hist"] = [samples[i]["drugs_hist"][i]] + samples[i]["drugs_hist"][:i]
        samples[i]['note'][i] = [1.0] * 768  # 防止free text泄漏
    return samples


def rec_prediction_eicu_fn(patient: Patient):
    samples = []
    for i in range(len(patient)):  # visit 次数
        visit: Visit = patient[i]
        conditions = visit.get_code_list(table="diagnosis")
        procedures = visit.get_code_list(table="physicalExam")
        drugs = visit.get_code_list(table="medication")


        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0: # 导致上面的过滤根本没用
            continue

        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "visit_id_hist": visit.visit_id,  # ['1']
                "drugs_hist": drugs,
                "conditions_raw": conditions,
                "procedures_raw": procedures,
                "labels": drugs,
            }
        )

    # exclude: patients with less than 2 visit
    if len(samples) < 1:  # [{visit 1},{visit 2}]
        return []

    samples[0]["conditions"] = [samples[0]["conditions"]]  # 这样visit对应的drug labels不用改;这里最少1个也行
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]]
    samples[0]["visit_id_hist"] = [samples[0]["visit_id_hist"]]

    for i in range(1, len(samples)):  # 第二次，到第N次，一个patient创建一个samples数据，迭代创建
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + [
            samples[i]["drugs_hist"]
        ]
        samples[i]["visit_id_hist"] = samples[i - 1]["visit_id_hist"] + [
            samples[i]["visit_id_hist"]
        ]

    # remove the target drug from the history
    for i in range(len(samples)):
        samples[i]["drugs_hist"][i] = ['<pad>', '<pad>', '<pad>']  # 去掉target
        # 时序对齐的padding
        samples[i]["drugs_hist"] = [samples[i]["drugs_hist"][i]] + samples[i]["drugs_hist"][:i]
    return samples


########## in-hospital mortality
def mor_prediction_miii_fn(patient: Patient):
    samples = []
    for i in range(len(patient) - 1):  # visit 次数
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]

        if next_visit.discharge_status not in [0, 1]:
            mortality_label = 0
        else:
            mortality_label = int(next_visit.discharge_status)

        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")

        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]

        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue

        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": drugs,  # 实际不需要使用，需要的其实是drugs_kg
                "visit_id_hist": [visit.visit_id],  # ['1']
                "drugs_hist": [drugs],
                "labels": mortality_label,
            }
        )

    if len(samples) < 1:  # [{},{}]
        return []

    samples[0]["conditions"] = samples[0]["conditions"]
    samples[0]["procedures"] = samples[0]["procedures"]
    samples[0]["drugs_hist"] = samples[0]["drugs_hist"]

    samples[0]["visit_id_hist"] = samples[0]["visit_id_hist"]

    for i in range(1, len(samples)):  # 第二次，到第N次，一个patient创建一个samples数据,这个samples是遍历很多次的数据
        samples[i]["conditions"] = samples[i - 1]["conditions"] + samples[i]["conditions"]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + samples[i]["procedures"]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + samples[i]["drugs_hist"]

        samples[i]["visit_id_hist"] = samples[i - 1]["visit_id_hist"] + samples[i]["visit_id_hist"]

    return samples


def mor_prediction_pic_fn(patient: Patient):
    samples = []
    for i in range(len(patient) - 1):  # visit 次数
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]

        if next_visit.discharge_status not in [0, 1]:
            mortality_label = 0
        else:
            mortality_label = int(next_visit.discharge_status)

        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="DIAGNOSES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")

        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]

        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue

        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": drugs,  # 实际不需要使用，需要的其实是drugs_kg
                "visit_id_hist": [visit.visit_id],  # ['1']
                "drugs_hist": [drugs],
                "labels": mortality_label,
            }
        )

    if len(samples) < 1:  # [{},{}]
        return []

    samples[0]["conditions"] = samples[0]["conditions"]
    samples[0]["procedures"] = samples[0]["procedures"]
    samples[0]["drugs_hist"] = samples[0]["drugs_hist"]

    samples[0]["visit_id_hist"] = samples[0]["visit_id_hist"]

    for i in range(1, len(samples)):  # 第二次，到第N次，一个patient创建一个samples数据,这个samples是遍历很多次的数据
        samples[i]["conditions"] = samples[i - 1]["conditions"] + samples[i]["conditions"]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + samples[i]["procedures"]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + samples[i]["drugs_hist"]

        samples[i]["visit_id_hist"] = samples[i - 1]["visit_id_hist"] + samples[i]["visit_id_hist"]

    return samples


def mor_prediction_miv_fn(patient: Patient):
    samples = []
    for i in range(len(patient) - 1):  # visit 次数
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]

        if next_visit.discharge_status not in [0, 1]:
            mortality_label = 0
        else:
            mortality_label = int(next_visit.discharge_status)

        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")


        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue

        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": drugs,  # 实际不需要使用，需要的其实是drugs_kg
                "visit_id_hist": [visit.visit_id],  # ['1']
                "drugs_hist": [drugs],
                "labels": mortality_label,
            }
        )

    if len(samples) < 2:  # [{},{}]
        return []

    samples[0]["conditions"] = samples[0]["conditions"]
    samples[0]["procedures"] = samples[0]["procedures"]
    samples[0]["drugs_hist"] = samples[0]["drugs_hist"]

    samples[0]["visit_id_hist"] = samples[0]["visit_id_hist"]

    for i in range(1, len(samples)):  # 第二次，到第N次，一个patient创建一个samples数据,这个samples是遍历很多次的数据
        samples[i]["conditions"] = samples[i - 1]["conditions"] + samples[i]["conditions"]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + samples[i]["procedures"]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + samples[i]["drugs_hist"]

        samples[i]["visit_id_hist"] = samples[i - 1]["visit_id_hist"] + samples[i]["visit_id_hist"]

    return samples


def mor_prediction_eicu_fn(patient: Patient):
    samples = []
    for i in range(len(patient) - 1):  # visit 次数
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]

        if next_visit.discharge_status not in ["Alive", "Expired"]:
            mortality_label = 0
        else:
            mortality_label = 0 if next_visit.discharge_status == "Alive" else 1


        conditions = visit.get_code_list(table="diagnosis")
        procedures = visit.get_code_list(table="physicalExam")
        drugs = visit.get_code_list(table="medication")

        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]

        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": drugs,  # 实际不需要使用，需要的其实是drugs_kg
                "visit_id_hist": [visit.visit_id],  # ['1']
                "drugs_hist": [drugs],
                "conditions_raw": conditions,
                "procedures_raw": procedures,
                "labels": mortality_label,
            }
        )

    if len(samples) < 2:  # [{},{}]
        return []

    samples[0]["conditions"] = samples[0]["conditions"]
    samples[0]["procedures"] = samples[0]["procedures"]
    samples[0]["drugs_hist"] = samples[0]["drugs_hist"]

    samples[0]["visit_id_hist"] = samples[0]["visit_id_hist"]

    for i in range(1, len(samples)):  # 第二次，到第N次，一个patient创建一个samples数据,这个samples是遍历很多次的数据
        samples[i]["conditions"] = samples[i - 1]["conditions"] + samples[i]["conditions"]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + samples[i]["procedures"]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + samples[i]["drugs_hist"]

        samples[i]["visit_id_hist"] = samples[i - 1]["visit_id_hist"] + samples[i]["visit_id_hist"]

    return samples





############ unify task
def los_prediction_miii_fn(patient: Patient):
    """Processes a single patient for the mortality prediction task.
    """
    samples = []
    for i in range(len(patient)):  # visit 次数
        visit: Visit = patient[i]

        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]

        # # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0: # bixu药在los之前
            continue

        # TODO: should also exclude visit with age < 18
        los_days = (visit.discharge_time - visit.encounter_time).days
        los_category = categorize_los(los_days)

        # exclude: visits without condition, procedure, or drug code

        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs_hist": [drugs],
                "visit_id_hist": [visit.visit_id],  # ['1']
                "drugs": drugs,  # used for diffusion
                "labels": los_category,
            }
        )

    if len(samples) < 1:  # [{},{}]； 这里why
        return []

    samples[0]["conditions"] = samples[0]["conditions"]
    samples[0]["procedures"] = samples[0]["procedures"]
    samples[0]["drugs_hist"] = samples[0]["drugs_hist"]
    samples[0]["visit_id_hist"] = samples[0]["visit_id_hist"]

    for i in range(1, len(samples)):  # 第二次，到第N次，一个patient创建一个samples数据,这个samples是遍历很多次的数据
        samples[i]["conditions"] = samples[i - 1]["conditions"] + samples[i]["conditions"]

        samples[i]["procedures"] = samples[i - 1]["procedures"] + samples[i]["procedures"]

        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + samples[i]["drugs_hist"]

        samples[i]["visit_id_hist"] = samples[i - 1]["visit_id_hist"] + samples[i]["visit_id_hist"]

    return samples

def los_prediction_pic_fn(patient: Patient):
    """Processes a single patient for the mortality prediction task.
    """
    samples = []
    for i in range(len(patient)):  # visit 次数
        visit: Visit = patient[i]

        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="DIAGNOSES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]

        # # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0: # bixu药在los之前
            continue

        # TODO: should also exclude visit with age < 18
        los_days = (visit.discharge_time - visit.encounter_time).days
        los_category = categorize_los(los_days)

        # exclude: visits without condition, procedure, or drug code

        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs_hist": [drugs],
                "visit_id_hist": [visit.visit_id],  # ['1']
                "drugs": drugs,  # used for diffusion
                "labels": los_category,
            }
        )

    if len(samples) < 1:  # [{},{}]； 这里why
        return []

    samples[0]["conditions"] = samples[0]["conditions"]
    samples[0]["procedures"] = samples[0]["procedures"]
    samples[0]["drugs_hist"] = samples[0]["drugs_hist"]

    samples[0]["visit_id_hist"] = samples[0]["visit_id_hist"]

    for i in range(1, len(samples)):  # 第二次，到第N次，一个patient创建一个samples数据,这个samples是遍历很多次的数据
        samples[i]["conditions"] = samples[i - 1]["conditions"] + samples[i]["conditions"]

        samples[i]["procedures"] = samples[i - 1]["procedures"] + samples[i]["procedures"]

        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + samples[i]["drugs_hist"]

        samples[i]["visit_id_hist"] = samples[i - 1]["visit_id_hist"] + samples[i]["visit_id_hist"]

    return samples

def los_prediction_miv_fn(patient: Patient):
    """Processes a single patient for the mortality prediction task.
    """
    samples = []
    for i in range(len(patient)):  # visit 次数
        visit: Visit = patient[i]

        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]

        if len(conditions) * len(procedures) * len(drugs) == 0: # for hyperrec， 必须要放在之前。
            continue

        # exclude: visits without condition, procedure, or drug code
        # TODO: should also exclude visit with age < 18
        los_days = (visit.discharge_time - visit.encounter_time).days
        los_category = categorize_los(los_days)

        # exclude: visits without condition, procedure, or drug code
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs_hist": [drugs],
                "visit_id_hist": [visit.visit_id],  # ['1']
                "drugs": drugs,  # used for diffusion
                "labels": los_category,
            }
        )

    if len(samples) < 2:  # [{},{}]； 这里why
        return []

    samples[0]["conditions"] = samples[0]["conditions"]
    samples[0]["procedures"] = samples[0]["procedures"]
    samples[0]["drugs_hist"] = samples[0]["drugs_hist"]
    samples[0]["visit_id_hist"] = samples[0]["visit_id_hist"]


    for i in range(1, len(samples)):  # 第二次，到第N次，一个patient创建一个samples数据,这个samples是遍历很多次的数据
        samples[i]["conditions"] = samples[i - 1]["conditions"] + samples[i]["conditions"]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + samples[i]["procedures"]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + samples[i]["drugs_hist"]
        samples[i]["visit_id_hist"] = samples[i - 1]["visit_id_hist"] + samples[i]["visit_id_hist"]

    return samples
def los_prediction_eicu_fn(patient: Patient):
    # 不知道length of stay是否有
    samples = []
    for i in range(len(patient)):  # visit 次数
        visit: Visit = patient[i]

        conditions = visit.get_code_list(table="diagnosis")
        procedures = visit.get_code_list(table="physicalExam")
        drugs = visit.get_code_list(table="medication")

        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0: # 在前后很重要
            continue

        los_days = (visit.discharge_time - visit.encounter_time).days
        los_category = categorize_los(los_days)

        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs_hist": [drugs],
                "visit_id_hist": [visit.visit_id],  # ['1']
                "drugs": drugs,  # used for diffusion
                "labels": los_category,
            }
        )

    if len(samples) < 2:  # [{},{}]； 这里why
        return []

    samples[0]["conditions"] = samples[0]["conditions"]
    samples[0]["procedures"] = samples[0]["procedures"]
    samples[0]["drugs_hist"] = samples[0]["drugs_hist"]

    samples[0]["visit_id_hist"] = samples[0]["visit_id_hist"]

    for i in range(1, len(samples)):  # 第二次，到第N次，一个patient创建一个samples数据,这个samples是遍历很多次的数据
        samples[i]["conditions"] = samples[i - 1]["conditions"] + samples[i]["conditions"]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + samples[i]["procedures"]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + samples[i]["drugs_hist"]
        samples[i]["visit_id_hist"] = samples[i - 1]["visit_id_hist"] + samples[i]["visit_id_hist"]

    return samples


def mor24_prediction_miii_fn(patient: Patient, future_time_interval=24):
    samples = []
    for i in range(len(patient)):  # visit 次数
        visit: Visit = patient[i]
        deathtime = patient.death_datetime
        encounter_time = visit.encounter_time

        if deathtime is None:
            lived_time = datetime.now() + timedelta(days=365 * 500) # Amx time
        else:
            lived_time = deathtime

        # 判断
        # if visit.discharge_status not in [0, 1]: # 就是None
        #     mortality_label = 0 # 0: alive, 1: dead
        # else:
        time_diff = (lived_time - encounter_time).total_seconds() / 3600
        mortality_label = int(time_diff < future_time_interval)

        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")

        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]


        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue

        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": drugs,  # 实际不需要使用，需要的其实是drugs_kg
                "visit_id_hist": [visit.visit_id],  # ['1']
                "drugs_hist": [drugs],
                "labels": mortality_label,

            }
        )

    if len(samples) < 1:  # [{},{}]
        return []

    samples[0]["conditions"] = samples[0]["conditions"]
    samples[0]["procedures"] = samples[0]["procedures"]
    samples[0]["drugs_hist"] = samples[0]["drugs_hist"]

    samples[0]["visit_id_hist"] = samples[0]["visit_id_hist"]

    for i in range(1, len(samples)):  # 第二次，到第N次，一个patient创建一个samples数据,这个samples是遍历很多次的数据
        samples[i]["conditions"] = samples[i - 1]["conditions"] + samples[i]["conditions"]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + samples[i]["procedures"]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + samples[i]["drugs_hist"]
        samples[i]["visit_id_hist"] = samples[i - 1]["visit_id_hist"] + samples[i]["visit_id_hist"]

    return samples


def mor24_prediction_pic_fn(patient: Patient, future_time_interval=24):
    samples = []
    for i in range(len(patient)):  # visit 次数
        visit: Visit = patient[i]
        deathtime = patient.death_datetime
        encounter_time = visit.encounter_time

        if deathtime is None:
            lived_time = datetime.now() + timedelta(days=365 * 500) # Amx time
        else:
            lived_time = deathtime

        # 判断
        # if visit.discharge_status not in [0, 1]: # 就是None
        #     mortality_label = 0 # 0: alive, 1: dead
        # else:
        time_diff = (lived_time - encounter_time).total_seconds() / 3600
        mortality_label = int(time_diff < future_time_interval)

        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="DIAGNOSES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")

        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]

        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": drugs,  # 实际不需要使用，需要的其实是drugs_kg
                "visit_id_hist": [visit.visit_id],  # ['1']
                "drugs_hist": [drugs],
                "labels": mortality_label,

            }
        )

    if len(samples) < 1:  # [{},{}]
        return []

    samples[0]["conditions"] = samples[0]["conditions"]
    samples[0]["procedures"] = samples[0]["procedures"]
    samples[0]["drugs_hist"] = samples[0]["drugs_hist"]

    samples[0]["visit_id_hist"] = samples[0]["visit_id_hist"]

    for i in range(1, len(samples)):  # 第二次，到第N次，一个patient创建一个samples数据,这个samples是遍历很多次的数据
        samples[i]["conditions"] = samples[i - 1]["conditions"] + samples[i]["conditions"]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + samples[i]["procedures"]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + samples[i]["drugs_hist"]
        samples[i]["visit_id_hist"] = samples[i - 1]["visit_id_hist"] + samples[i]["visit_id_hist"]

    return samples


def mor24_prediction_miv_fn(patient: Patient, future_time_interval=24):
    samples = []
    for i in range(len(patient)):  # visit 次数
        visit: Visit = patient[i]
        deathtime = patient.death_datetime
        encounter_time = visit.encounter_time

        if deathtime is None:
            lived_time = datetime.now() + timedelta(days=365 * 500) # Amx time
        else:
            lived_time = deathtime

        # 判断
        # if visit.discharge_status not in [0, 1]: # 就是None
        #     mortality_label = 0 # 0: alive, 1: dead
        # else:
        time_diff = (lived_time - encounter_time).total_seconds() / 3600
        mortality_label = int(time_diff < future_time_interval)

        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")

        # ATC 3 level

        drugs = [drug[:4] for drug in drugs]

        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue

        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": drugs,  # 实际不需要使用，需要的其实是drugs_kg
                "visit_id_hist": [visit.visit_id],  # ['1']
                "drugs_hist": [drugs],
                "labels": mortality_label,
            }
        )

    if len(samples) < 2:  # [{},{}]
        return []

    samples[0]["conditions"] = samples[0]["conditions"]
    samples[0]["procedures"] = samples[0]["procedures"]
    samples[0]["drugs_hist"] = samples[0]["drugs_hist"]

    samples[0]["visit_id_hist"] = samples[0]["visit_id_hist"]

    for i in range(1, len(samples)):  # 第二次，到第N次，一个patient创建一个samples数据,这个samples是遍历很多次的数据
        samples[i]["conditions"] = samples[i - 1]["conditions"] + samples[i]["conditions"]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + samples[i]["procedures"]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + samples[i]["drugs_hist"]
        samples[i]["visit_id_hist"] = samples[i - 1]["visit_id_hist"] + samples[i]["visit_id_hist"]

    return samples


def mor24_prediction_eicu_fn(patient: Patient, future_time_interval=24):
    samples = []
    for i in range(len(patient)):  # visit 次数
        visit: Visit = patient[i]
        deathtime = patient.death_datetime
        encounter_time = visit.encounter_time

        if deathtime is None:
            lived_time = datetime.now() + timedelta(days=365 * 500) # Amx time
        else:
            lived_time = deathtime

        # 判断
        # if visit.discharge_status not in [0, 1]: # 就是None
        #     mortality_label = 0 # 0: alive, 1: dead
        # else:
        time_diff = (lived_time - encounter_time).total_seconds() / 3600
        mortality_label = int(time_diff < future_time_interval)

        conditions = visit.get_code_list(table="diagnosis")
        procedures = visit.get_code_list(table="physicalExam")
        drugs = visit.get_code_list(table="medication")


        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue

        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": drugs,  # 实际不需要使用，需要的其实是drugs_kg
                "visit_id_hist": [visit.visit_id],  # ['1']
                "drugs_hist": [drugs],
                "labels": mortality_label,

            }
        )

    if len(samples) < 2:  # [{},{}]
        return []

    samples[0]["conditions"] = samples[0]["conditions"]
    samples[0]["procedures"] = samples[0]["procedures"]
    samples[0]["drugs_hist"] = samples[0]["drugs_hist"]

    samples[0]["visit_id_hist"] = samples[0]["visit_id_hist"]

    for i in range(1, len(samples)):  # 第二次，到第N次，一个patient创建一个samples数据,这个samples是遍历很多次的数据
        samples[i]["conditions"] = samples[i - 1]["conditions"] + samples[i]["conditions"]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + samples[i]["procedures"]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + samples[i]["drugs_hist"]
        samples[i]["visit_id_hist"] = samples[i - 1]["visit_id_hist"] + samples[i]["visit_id_hist"]

    return samples




def read_prediction_miii_fn(patient: Patient, time_window=15):
    samples = []

    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]

        # get time difference between current visit and next visit
        time_diff = (next_visit.encounter_time - visit.encounter_time).days
        readmission_label = 1 if time_diff < time_window else 0

        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")

        drugs = [drug[:4] for drug in drugs]

        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": drugs,  # 实际不需要使用，需要的其实是drugs_kg，为啥这里需要套起来
                "visit_id_hist": [visit.visit_id],  # ['1']
                "drugs_hist": [drugs],
                "labels": readmission_label,
            }
        )
    #
    if len(samples) < 1:  # [{},{}]
        return []

    samples[0]["conditions"] = samples[0]["conditions"]
    samples[0]["procedures"] = samples[0]["procedures"]
    samples[0]["drugs_hist"] = samples[0]["drugs_hist"]
    samples[0]["visit_id_hist"] = samples[0]["visit_id_hist"]

    for i in range(1, len(samples)):  # 第二次，到第N次，一个patient创建一个samples数据,这个samples是遍历很多次的数据
        samples[i]["conditions"] = samples[i - 1]["conditions"] + samples[i]["conditions"]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + samples[i]["procedures"]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + samples[i]["drugs_hist"]
        samples[i]["visit_id_hist"] = samples[i - 1]["visit_id_hist"] + samples[i]["visit_id_hist"]

    # no cohort selection
    return samples


def read_prediction_pic_fn(patient: Patient, time_window=15):
    samples = []

    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]

        # get time difference between current visit and next visit
        time_diff = (next_visit.encounter_time - visit.encounter_time).days
        readmission_label = 1 if time_diff < time_window else 0

        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="DIAGNOSES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")

        drugs = [drug[:4] for drug in drugs]

        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": drugs,  # 实际不需要使用，需要的其实是drugs_kg，为啥这里需要套起来
                "visit_id_hist": [visit.visit_id],  # ['1']
                "drugs_hist": [drugs],
                "labels": readmission_label,
            }
        )
    #
    if len(samples) < 1:  # [{},{}]
        return []

    samples[0]["conditions"] = samples[0]["conditions"]
    samples[0]["procedures"] = samples[0]["procedures"]
    samples[0]["drugs_hist"] = samples[0]["drugs_hist"]
    samples[0]["visit_id_hist"] = samples[0]["visit_id_hist"]

    for i in range(1, len(samples)):  # 第二次，到第N次，一个patient创建一个samples数据,这个samples是遍历很多次的数据
        samples[i]["conditions"] = samples[i - 1]["conditions"] + samples[i]["conditions"]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + samples[i]["procedures"]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + samples[i]["drugs_hist"]
        samples[i]["visit_id_hist"] = samples[i - 1]["visit_id_hist"] + samples[i]["visit_id_hist"]
    # no cohort selection
    return samples



def read_prediction_miv_fn(patient: Patient, time_window=15):
    samples = []

    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]

        # get time difference between current visit and next visit
        time_diff = (next_visit.encounter_time - visit.encounter_time).days
        readmission_label = 1 if time_diff < time_window else 0

        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")

        drugs = [drug[:4] for drug in drugs]

        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": drugs,  # 实际不需要使用，需要的其实是drugs_kg，为啥这里需要套起来
                "visit_id_hist": [visit.visit_id],  # ['1']
                "drugs_hist": [drugs],
                "labels": readmission_label,
            }
        )
    #
    if len(samples) < 2:  # [{},{}]
        return []

    samples[0]["conditions"] = samples[0]["conditions"]
    samples[0]["procedures"] = samples[0]["procedures"]
    samples[0]["drugs_hist"] = samples[0]["drugs_hist"]
    samples[0]["visit_id_hist"] = samples[0]["visit_id_hist"]

    for i in range(1, len(samples)):  # 第二次，到第N次，一个patient创建一个samples数据,这个samples是遍历很多次的数据
        samples[i]["conditions"] = samples[i - 1]["conditions"] + samples[i]["conditions"]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + samples[i]["procedures"]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + samples[i]["drugs_hist"]
        samples[i]["visit_id_hist"] = samples[i - 1]["visit_id_hist"] + samples[i]["visit_id_hist"]

    # no cohort selection
    return samples


def read_prediction_eicu_fn(patient: Patient, time_window=15):
    samples = []

    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]

        # get time difference between current visit and next visit
        time_diff = (next_visit.encounter_time - visit.encounter_time).days
        readmission_label = 1 if time_diff < time_window else 0

        conditions = visit.get_code_list(table="diagnosis")
        procedures = visit.get_code_list(table="physicalExam")
        drugs = visit.get_code_list(table="medication")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": drugs,  # 实际不需要使用，需要的其实是drugs_kg，为啥这里需要套起来
                "visit_id_hist": [visit.visit_id],  # ['1']
                "drugs_hist": [drugs],

                "labels": readmission_label,
            }
        )
    #
    if len(samples) < 2:  # [{},{}]
        return []

    samples[0]["conditions"] = samples[0]["conditions"]
    samples[0]["procedures"] = samples[0]["procedures"]
    samples[0]["drugs_hist"] = samples[0]["drugs_hist"]
    samples[0]["visit_id_hist"] = samples[0]["visit_id_hist"]

    for i in range(1, len(samples)):  # 第二次，到第N次，一个patient创建一个samples数据,这个samples是遍历很多次的数据
        samples[i]["conditions"] = samples[i - 1]["conditions"] + samples[i]["conditions"]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + samples[i]["procedures"]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + samples[i]["drugs_hist"]
        samples[i]["visit_id_hist"] = samples[i - 1]["visit_id_hist"] + samples[i]["visit_id_hist"]

    # no cohort selection
    return samples



def get_task_fn(config):
    dataset = config['DATASET']
    task = config['TASK']
    all_fn = {
        'MIII': {
            'PHE_fn': phe_prediction_miii_fn,
            'DIAG_fn': None,
            'LOS_fn': los_prediction_miii_fn,
            'MOR_fn': mor24_prediction_miii_fn,#mor_prediction_miii_fn, # 下次补上
            'IHM_fn': mor_prediction_miii_fn,
            'REA_fn': read_prediction_miii_fn,
            'REC_fn': rec_prediction_miii_fn,
        },
        'MIV': {
            'PHE_fn': None,
            'DIAG_fn': None,
            'LOS_fn': los_prediction_miv_fn,
            'MOR_fn': mor24_prediction_miv_fn,  # 下次补上
            'IHM_fn': mor_prediction_miv_fn,
            'REA_fn': read_prediction_miv_fn,
            'REC_fn': rec_prediction_miv_fn,
        },
        'MIV-Note': {
            'PHE_fn': phe_prediction_miv_note_fn,
            'DIAG_fn': None,
            'LOS_fn': None,
            'MOR_fn': None,
            'REA_fn': None,
            'REC_fn': rec_prediction_miv_note_fn,
        },
        'eICU': {
            'PHE_fn': phe_prediction_eicu_fn,
            'DIAG_fn': None,
            'LOS_fn': los_prediction_eicu_fn,
            'MOR_fn': mor24_prediction_eicu_fn,
            'IHM_fn': mor_prediction_eicu_fn,
            'REA_fn': read_prediction_eicu_fn,
            'REC_fn': rec_prediction_eicu_fn,
        },
        'EHR-SHOT': {
            'PHE_fn': None,
            'DIAG_fn': diag_prediction_ehrshot_fn,
            'LOS_fn': None,
            'MOR_fn': None,
            'REA_fn': None,
            'REC_fn': None,
        },
        'OMOP': {
            'PHE_fn': None,
            'DIAG_fn': diag_prediction_omop_fn,
            'LOS_fn': None,
            'MOR_fn': None,
            'REA_fn': None,
            'REC_fn': None,
        },
        'PIC': {
            'PHE_fn': None,
            'DIAG_fn':None,
            'LOS_fn': None,
            'MOR_fn': None,
            'REA_fn': None,
            'REC_fn': None,
        },
        'OMIX': {
            'PHE_fn': None,
            'DIAG_fn':None,
            'LOS_fn': None,
            'MOR_fn': None,
            'REA_fn': None,
            'REC_fn': None,
        }
    }
    if dataset in ['MIII', 'MIV', 'MIV-Note', 'eICU', 'EHR-SHOT', 'OMOP', 'PIC', 'OMIX']:
        task_fn = all_fn[dataset][task + '_fn']
    else:
        task_fn = None#'others'
    # if task_fn is None:
    #     raise ValueError("Task not implementation!")
    if task in ['PHE', 'DIAG', 'REC', 'MULTIPLE']:
        mode = 'multilabel'
    elif task in ['LOS', 'SINGLE']:
        mode = 'multiclass'
    elif task in ['MOR', 'REA', 'IHM']:
        mode = 'binary'
    elif task in ['SUMMARY']:
        mode = 'text'
    else:
        raise ValueError("Mode not supported!")

    if task in ['PHE', 'DIAG', 'REC', 'LOS', 'MOR', 'REA', 'IHM']: # ehr PREDICTION
        task_mode ='ehr'
    elif task in ['SUMMARY']: # free text生成
        task_mode = 'summary'
    elif task in ['MULTIPLE']: # 多项选择
        task_mode = 'mqa'
    elif task in ['SINGLE']: # 单项选择
        task_mode = 'qa'
    else:
        raise ValueError("Task mode not supported!")

    return task_fn, mode, task_mode
