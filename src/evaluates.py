# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : evaluate.py
# Time       ：24/4/2025 11:39 am
# Author     ：Any
# version    ：python 
# Description：这个要搞定。PPO训练好后，要读取对应的model.
"""
import json
import ray
import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from models import AgentPipeline
from utils import get_emb_model, load_graph_data,get_mode, get_normalize_text, get_metrics_fn, locate_answer
from dataloader import get_dataloader, get_special_input
from instructions_template import task_templates
# from agent_low import AgentLowOur
# from agent_top import AgentTopOur
from dataset import SimpleDataset, get_llm_model
from models import KARE, MedRAG, LightRAG, CoT, PureLLM
from langchain.schema.runnable import RunnableConfig
from tqdm import tqdm  # 导入tqdm
from utils import get_results_parallel

def evaluate_generation_flask(model_path, data_path, config, output_path,task_mode='ehr', run_config:RunnableConfig=RunnableConfig(llm={}), topk=1, p_grouped=None):
    print("Current model path!")
    # 读取权重
    llm = None
    embedding_model = None

    # 读取数据
    dataset = SimpleDataset(data_path)
    collate_fn = get_special_input(config)
    dataloader = get_dataloader(dataset, batch_size=config['BATCH'], shuffle=False, drop_last=True, collate_fn=collate_fn)

    # 正常的index
    root_to_dir = "/hpc2hdd/home/xxxs349/xxxc/RAGHealth/data/ready" # 这个路径都是公用的
    graph, node_data, metapaths_dic = load_graph_data(root_to_dir)
    metapaths_dic = {key: value for key, value in metapaths_dic.items() if key in map(str, range(0, 10))} # 和create_sft中保持一致。

    meta_paths = [metapaths_dic[i]['raw-meta-path'] for i in metapaths_dic]


    # 分布式设定
    print('Agent Done! No need load graph. But you need to run the flask server!')
    gpu_count = torch.cuda.device_count()
    ray.init(num_gpus=gpu_count) # 用4个gpu，供所有actor可见， 最好和batch相同或者可以被batch整除
    # # 创建类的实例, 不传入gpus则默认在CPU上面运行，
    ray_workers = [AgentPipeline.remote(llm, embedding_model, metapaths_dic, ratio=config['META_RATIO'], topk=config['TOPK'], config=config, eval_path=model_path) for i in range(gpu_count)]
    # 创建 4 个 Actor 实例，分布在 4 个 GPU 上
    print("RAY USE GPU IDS", ray.get_gpu_ids())

    # 调用类方法
    no_preds = 0
    with open(output_path, 'w') as json_file:  # 使用 'w' 模式写入数据
        pred_norms, golden_norms, query_norms = [], [], []
        # for index, batch in enumerate(dataloader):
        total_batches = len(dataloader)  # 获取总批次数量（若dataloader支持）
        for index, batch in enumerate(tqdm(dataloader, desc="Processing batches", total=total_batches)):
            # 创建actor实例
            # futures = [ray_workers[index_ray].predict.remote(
            #     query=quer,
            #     ground_truth=gold, run_config=run_config, topk=topk)
            #            for index_ray, quer, gold in zip(list(range(config['BATCH'])), batch['query'], batch['groundtruth'])]
            #
            # # 每个actor进行action
            # results = ray.get(futures) # 一个batch的数据

            futures = []
            querys, ground_truths = batch['query'], batch['groundtruth']
            mini_batch_size = config['BATCH'] // gpu_count
            for i in range(0, len(querys), mini_batch_size):
                # 提取mini-batch并准备输入
                mini_batch_querys = querys[i:i + mini_batch_size]
                mini_batch_ground_truths = ground_truths[i:i + mini_batch_size]

                # 分配给worker（循环使用）
                worker_idx = i // mini_batch_size % len(ray_workers)
                futures.append(ray_workers[worker_idx].batch_predict.remote(mini_batch_querys, mini_batch_ground_truths))
            # 获取所有worker的结果
            results = get_results_parallel(futures)  # ray.get(futures)

            for mini_batch_data in results:
                for batch_data in mini_batch_data:
                    query, ground_truth, final_answer, golden_paths, _ = batch_data
                    # 假设这是你的字典数据
                    data_dict = {
                        "golden_paths": golden_paths, # 如果golden_path为空，则证明在setting内没有找到答案。
                        "query": query,
                        "predict": final_answer,
                        "groundtruth": ground_truth
                    }
                    # 数值
                    final_answer = locate_answer(final_answer)
                    pred_norm, golden_norm, no_pred = get_normalize_text(final_answer, ground_truth, config)
                    # print("AAAAA", pred_norm, golden_norm)
                    # print("="*8)
                    no_preds += no_pred
                    pred_norms.append(pred_norm)
                    golden_norms.append(golden_norm)
                    query_norms.append(query) # 这个说不定要改
                    data_dict['predict_value'] = pred_norm
                    data_dict['groundtruth_value'] = golden_norm

                    json.dump(data_dict, json_file)
                    json_file.write('\n')  # 每个批次结果换行

    # 计算metrics
    print("You have finished in the pipeline!")
    print("No preds: ", no_preds)
    # 定义指标
    mode, metrics = get_mode(config)
    metrics_fn = get_metrics_fn(mode)
    if mode in ['qa', 'summary', 'mqa']:
        scores = metrics_fn(np.array(golden_norms), np.array(pred_norms), metrics=metrics, aux_data={'sources': query_norms})
    else:
        scores = metrics_fn(np.array(golden_norms), np.array(pred_norms), metrics=metrics, patient_ids=p_grouped[0], aux_data={'p_grouped' :p_grouped[1]})
    print("Final metrics. ", scores)




def evaluate_generation_think_flask(model_path, data_path, config, output_path, task_mode='ehr',run_config:RunnableConfig=RunnableConfig(llm={}), topk=1, p_grouped=None):
    # for baseline
    # print("Current model path!")
    # 读取权重
    llm = None
    embedding_model = None

    # 读取数据
    dataset = SimpleDataset(data_path)
    collate_fn = get_special_input(config)
    dataloader = get_dataloader(dataset, batch_size=config['BATCH'], shuffle=False, drop_last=True, collate_fn=collate_fn)

    # 正常的index
    root_to_dir = "/hpc2hdd/home/xxxs349/xxxc/RAGHealth/data/ready" # 这个路径都是公用的
    graph, node_data, metapaths_dic = load_graph_data(root_to_dir)
    meta_paths = [metapaths_dic[i]['raw-meta-path'] for i in metapaths_dic]


    # 分布式设定
    print('Agent Done! No need load graph. But you need to run the flask server!')
    gpu_count = torch.cuda.device_count()

    ray.init(num_gpus=gpu_count) # 用4个gpu，供所有actor可见， 最好和batch相同或者可以被batch整除
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
            CoT.remote(llm, embedding_model,meta_paths, topk=config['TOPK'], config=config) for i in range(gpu_count)]
    else:
        raise ValueError("Invalid model name. Choose from ['KARE', 'MedRAG', 'LightRAG']")
    print('Model Done! Please confirm the ray workers are created successfully. If not, please check the GPU setting.')

    # 调用类方法
    no_preds = 0
    with open(output_path, 'w') as json_file:  # 使用 'w' 模式写入数据
        pred_norms, golden_norms,query_norms = [], [],[]
        # for index, batch in enumerate(dataloader):
        total_batches = len(dataloader)  # 获取总批次数量（若dataloader支持）
        for index, batch in enumerate(tqdm(dataloader, desc="Processing batches", total=total_batches)):
            # # 创建actor实例
            # futures = [ray_workers[ray_index].predict.remote(
            #     query=quer,
            #     ground_truth=gold, run_config=run_config, topk=topk)
            #            for ray_index, quer, gold in zip(list(range(config['BATCH'])), batch['query'], batch['groundtruth'])]
            #
            # # 每个actor进行action
            # results = ray.get(futures) # 一个batch的数据

            futures = []
            querys, ground_truths = batch['query'], batch['groundtruth']
            mini_batch_size = config['BATCH'] // gpu_count
            for i in range(0, len(querys), mini_batch_size):
                # 提取mini-batch并准备输入
                mini_batch_querys = querys[i:i + mini_batch_size]
                mini_batch_ground_truths = ground_truths[i:i + mini_batch_size]

                # 分配给worker（循环使用）
                worker_idx = i // mini_batch_size % len(ray_workers)
                futures.append(ray_workers[worker_idx].batch_predict.remote(mini_batch_querys, mini_batch_ground_truths))
            # 获取所有worker的结果
            results = get_results_parallel(futures)  # ray.get(futures)


            for mini_batch_data in results:
                for batch_data in mini_batch_data:
                    query,think_chain,final_answer, ground_truth,decision = batch_data
                    # 假设这是你的字典数据
                    data_dict = {
                        "think_chain": think_chain, # 如果golden_path为空，则证明在setting内没有找到答案。
                        "query": query,
                        "predict": final_answer,
                        "groundtruth": ground_truth,
                        "decision": decision
                    }
                    # 数值
                    final_answer = locate_answer(final_answer)
                    pred_norm, golden_norm, no_pred = get_normalize_text(final_answer, ground_truth, config)
                    # print("AAAAA", pred_norm, golden_norm)
                    # print("="*8)
                    no_preds += no_pred
                    pred_norms.append(pred_norm)
                    golden_norms.append(golden_norm)
                    query_norms.append(query) # 这个说不定要改

                    data_dict['predict_value'] = pred_norm
                    data_dict['groundtruth_value'] = golden_norm

                    json.dump(data_dict, json_file)
                    json_file.write('\n')  # 每个批次结果换行

                # if index > 1: # 仅使用部分的数据。
                #     break

    # 计算metrics
    print("You have finished in the pipeline!")
    print("No preds: ", no_preds)


    # # 如果突然断掉， 在跑了很长时间的情况下
    # golden_norms = []
    # query_norms = []
    # pred_norms = []
    # with open(output_path, 'r') as json_file:
    #     for line in json_file:
    #         try:
    #             data_dict = json.loads(line)
    #             golden_norms.append(data_dict['groundtruth_value'])
    #             query_norms.append(data_dict['query'])
    #             pred_norms.append(data_dict['predict_value'])
    #         except:
    #             continue
    # print('p_grouped', p_grouped)

    # 定义指标
    mode, metrics = get_mode(config)
    metrics_fn = get_metrics_fn(mode)
    if mode in ['qa', 'summary', 'mqa']:
        scores = metrics_fn(np.array(golden_norms), np.array(pred_norms), metrics=metrics, aux_data={'sources': query_norms})
    else:
        scores = metrics_fn(np.array(golden_norms), np.array(pred_norms), metrics=metrics,patient_ids=p_grouped[0], aux_data={'p_grouped' :p_grouped[1]})
    print("Final metrics. ", scores)






def evaluate_generation_infer_flask(model_path, data_path, config, output_path, task_mode='ehr', run_config:RunnableConfig=RunnableConfig(llm={}), topk=1, p_grouped=None):
    print("Current model path!")
    # 读取权重
    llm = None
    embedding_model = None

    # 读取数据
    dataset = SimpleDataset(data_path)
    collate_fn = get_special_input(config)
    dataloader = get_dataloader(dataset, batch_size=config['BATCH'], shuffle=False, drop_last=False, collate_fn=collate_fn)


    # 正常的index
    root_to_dir = "/hpc2hdd/home/xxxs349/xxxc/RAGHealth/data/ready" # 这个路径都是公用的
    graph, node_data, metapaths_dic = load_graph_data(root_to_dir)
    meta_paths = [metapaths_dic[i]['raw-meta-path'] for i in metapaths_dic]



    # 分布式设定
    print('Agent Done! No need load graph. But you need to run the flask server!')
    gpu_count = torch.cuda.device_count()

    ray.init(num_gpus=gpu_count) # 用4个gpu，供所有actor可见， 最好和batch相同或者可以被batch整除
    # # 创建类的实例, 不传入gpus则默认在CPU上面运行，
    if config['MODEL'] in ['meditron-7B', 'qwen25-32B', 'deepseekr1-7B', 'biomistral-7B', 'LLAMA3-3B', 'LLAMA3-8B', 'qwen25-7B', 'qwq-32B']:
        ray_workers = [
            PureLLM.remote(llm, embedding_model, meta_paths, topk=config['TOPK'], config=config) for i in range(gpu_count)]
    else:
        raise ValueError("Invalid model name.")
    print('Model Done! Please confirm the ray workers are created successfully. If not, please check the GPU setting.')

    # 调用类方法
    with open(output_path, 'w') as json_file:  # 使用 'w' 模式写入数据
        pred_norms, golden_norms, query_norms = [], [], []
        no_preds = 0
        total_batches = len(dataloader)  # 获取总批次数量（若dataloader支持）
        for index, batch in enumerate(tqdm(dataloader, desc="Processing batches", total=total_batches)):
            # 创建actor实例
            # futures = [ray_workers[ray_index].predict.remote(
            #     query=quer,
            #     ground_truth=gold, run_config=run_config, topk=topk)
            #            for ray_index, quer, gold in zip(list(range(config['BATCH'])),batch['query'], batch['groundtruth'])]
            #
            # # 每个actor进行action
            # results = ray.get(futures) # 一个batch的数据

            futures = []
            querys, ground_truths = batch['query'], batch['groundtruth']
            mini_batch_size = config['BATCH'] // gpu_count
            print("AAAAAAAA", len(batch),len(querys), mini_batch_size)

            for i in range(0, len(querys), mini_batch_size):
                # 提取mini-batch并准备输入
                mini_batch_querys = querys[i:i + mini_batch_size]
                mini_batch_ground_truths = ground_truths[i:i + mini_batch_size]

                # 分配给worker（循环使用）
                worker_idx = i // mini_batch_size % len(ray_workers)
                futures.append(ray_workers[worker_idx].batch_predict.remote(mini_batch_querys, mini_batch_ground_truths))
            # 获取所有worker的结果
            results = get_results_parallel(futures)  # ray.get(futures)

            for mini_batch_data in results:
                for batch_data in mini_batch_data:
                    query, final_answer, ground_truth = batch_data
                    # 假设这是你的字典数据
                    data_dict = {
                        "query": query,
                        "predict": final_answer,
                        "groundtruth": ground_truth,
                    }
                    # 数值
                    final_answer = locate_answer(final_answer)
                    pred_norm, golden_norm, no_pred = get_normalize_text(final_answer, ground_truth, config)
                    no_preds += no_pred
                    # print("AAAAA", pred_norm, golden_norm)
                    # print("="*8)
                    pred_norms.append(pred_norm)
                    golden_norms.append(golden_norm)
                    query_norms.append(query) # 这个说不定要改

                    data_dict['predict_value'] = pred_norm
                    data_dict['groundtruth_value'] = golden_norm

                    json.dump(data_dict, json_file)
                    json_file.write('\n')  # 每个批次结果换行

    # 计算metrics
    print("You have finished in the pipeline!")
    print("No preds: ", no_preds)
    #
    # # # 如果突然断掉， 在跑了很长时间的情况下
    # golden_norms = []
    # query_norms = []
    # pred_norms = []
    # with open(output_path, 'r') as json_file:
    #     for line in json_file:
    #         try:
    #             data_dict = json.loads(line)
    #             golden_norms.append(data_dict['groundtruth_value'])
    #             query_norms.append(data_dict['query'])
    #             pred_norms.append(data_dict['predict_value'])
    #         except:
    #             continue
    # print('p_grouped', p_grouped)

    # 定义指标
    mode, metrics = get_mode(config)
    metrics_fn = get_metrics_fn(mode)
    if mode in ['qa', 'summary', 'mqa']:
        scores = metrics_fn(np.array(golden_norms), np.array(pred_norms), metrics=metrics,aux_data={'sources': query_norms})
    else:
        scores = metrics_fn(np.array(golden_norms), np.array(pred_norms), metrics=metrics, patient_ids=p_grouped[0], aux_data={'p_grouped' :p_grouped[1]})
    print("Final metrics. ", scores)





def evaluate_generation_flash_flask(model_path, data_path, config, output_path, task_mode='ehr', run_config:RunnableConfig=RunnableConfig(llm={}), topk=1, p_grouped=None):
    print("Current model path!")
    # 读取数据
    import pickle
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    querys, results, answers = data['query'], data['output'], data['gold_answer']

    # 数值
    no_preds = 0
    pred_norms, golden_norms, query_norms = [], [], []
    for query, final_answer, ground_truth in zip(querys, results, answers):
        final_answer = locate_answer(final_answer)

        # match = re.search(r'<answer>(.*?)<\/answer>', final_answer)
        #
        # if match:
        #     final_answer = match.group(1)
        # else:
        #     final_answer = final_answer

        # print("DDDDDD", final_answer)
        pred_norm, golden_norm, no_pred = get_normalize_text(final_answer, ground_truth, config)
        no_preds += no_pred
        # print("AAAAA", pred_norm, golden_norm)
        # print("="*8)
        pred_norms.append(pred_norm)
        golden_norms.append(golden_norm)
        query_norms.append(query)  # 这个说不定要


    print("You have finished in the pipeline!")
    print("No preds: ", no_preds)

    #
    # pred_norms, golden_norms, query_norms = [], [], []
    # with open(output_path, 'r') as json_file:
    #     for line in json_file:
    #         try:
    #             data_dict = json.loads(line)
    #             golden_norms.append(data_dict['groundtruth_value'])
    #             query_norms.append(data_dict['query'])
    #             pred_norms.append(data_dict['predict_value'])
    #         except:
    #             continue


    # 定义指标
    mode, metrics = get_mode(config)
    metrics_fn = get_metrics_fn(mode)
    if mode in ['qa', 'summary', 'mqa']:
        scores = metrics_fn(np.array(golden_norms), np.array(pred_norms), metrics=metrics,aux_data={'sources': query_norms})
    else:
        scores = metrics_fn(np.array(golden_norms), np.array(pred_norms), metrics=metrics, patient_ids=p_grouped[0], aux_data={'p_grouped' :p_grouped[1]})
    print("Final metrics. ", scores)





if __name__ == '__main__':
    from config import config
    # 这里evaluate必须要进行专门的设定。在main函数中运行。
    model_path = '/hpc2hdd/home/xxxs349/xxxc/RAGHealth/LLaMA-Factory/output/llama3_lora_reward'
    data_path = '/hpc2hdd/home/xxxs349/xxxc/RAGHealth/data/MOR/MIII/processed/pure_path_test.json'
    output_path = '/hpc2hdd/home/xxxs349/xxxc/RAGHealth/data/MOR/MIII/processed/test_output.json'
    evaluate_generation_flask(model_path=model_path, data_path=data_path, config=config, output_path=output_path)

