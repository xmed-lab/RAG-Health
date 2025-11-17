# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : models.py
# Time       ：14/3/2025 4:56 pm
# Author     ：Any
# version    ：python 
# Description： """

from langchain.schema.runnable import RunnableConfig
import asyncio
from typing import List, Dict, Any
from agent_top import AgentTop
from agent_low import AgentLow
import ray
import torch
import os
import re
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from baseline.kare import AgentKARE
from baseline.medrag import AgentMedRAG
from baseline.cot import AgentCoT
from baseline.lightrag import AgentLight
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from vllm import LLM, SamplingParams
from accelerate import dispatch_model, infer_auto_device_map
from torch.nn.parallel import DataParallel  # 数据并行（备用方案）
import json
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer, BatchEncoding
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import pipeline as Pipeline
from langchain_core.prompts import PromptTemplate
from utils import get_emb_model, get_llm_model # 不要葱dataset中引入，不然会有circular import
from instructions_template import purellm_template
from typing import Optional, Tuple, Union, List

# def get_model_workers(llm, embedding_model, metapaths_dic, config):
#     if config['MODEL'] == 'ours':
#         ray_workers = [AgentPipeline.remote(llm, embedding_model, metapaths_dic, ratio=config['META_RATIO'], topk=config['TOPK'],
#                                  config=config) for i in range(config['BATCH'])]
#         return ray_workers
#     elif config['MODEL'] == 'KARE':
#         pass
#     elif config['MODEL'] == 'MedRAG':
#         pass
#     elif config['MODEL'] == 'LightRAG':
#         pass
#     else:
#         raise ValueError("Invalid model name. Choose from ['ours', 'KARE', 'MedRAG', 'LightRAG']")


@ray.remote(num_gpus=1,num_cpus=2) #和 create_sft_data_flask一起使用, 每个actor使用一个gpu; 不能占用过多GPU
class AgentPipeline:
    def __init__(self, llm: Any, embedding_model: Any, metapaths: List[Dict[str, str]], ratio: float = 0.3, topk: int=3, max_iter=5, config=None, eval_path=None):
        """
        初始化 Pipeline，包含 AgentTop 和 AgentLow。

        Args:
            llm: 语言模型实例（用于 AgentTop 和 AgentLow）。
            embedding_model: 文本向量化模型（用于 AgentLow）。
            metapaths: 元路径列表，包含 name 和 description。
            ratio: 选择 meta-path 的比例（默认 0.3）。
        """
        print(f"CUDA Devices: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        self.device = torch.cuda.current_device() # 这样记录device也不知道行不行xs

        llm, embedding_model = self.flask_llm_emb(config, eval_path) # 和flask一起使用
        # 初始化 AgentLow，传入 LLM 和 embedding_model
        self.agent_low = AgentLow(embedding_model, llm, TOPK=topk) # 暂定不妨model，因为不更新

        # 初始化 AgentTop，传入 LLM 和 AgentLow
        self.agent_top = AgentTop(llm, metapaths, self.agent_low, ratio, max_iterations=max_iter, config=config)
        print("Initialization Done!")

    def flask_llm_emb(self,config, eval_path=None):
        if eval_path is None:
            llm = get_llm_model(config, config['T_LLM'], api_base=None, path=config['T_LLM_PATH'],device=self.device)
        else:
            llm = get_llm_model(config, config['T_LLM'], api_base=None, path=eval_path, device=self.device)
        emb_model = get_emb_model(config, device=self.device)
        return llm, emb_model


    def load_graph(self, graph: Any):
        """
        加载图数据到 AgentLow。

        Args:
            graph: DGL 异构图实例。
        """
        self.agent_low.load_graph(graph)
        print('Agent Load Graph Done!')

    def add_text_chunks(self, chunks: List[str]):
        """
        添加文本块到 AgentLow 的 Naive RAG 存储中。

        Args:
            chunks: 文本块列表。
        """
        self.agent_low.add_text_chunks(chunks)
        print('Agent Index Text Done!')




    def batch_run(self, queries: List[str], ground_truths: List[str], run_config:RunnableConfig=RunnableConfig(llm={}), topk=1) -> List[str]:
        """
        执行批量的 Pipeline 流程：
        1. 使用 AgentTop 进行任务分解和决策。
        2. 使用 AgentLow 进行检索和总结。
        3. 生成最终答案。

        Args:
            queries: 用户查询文本列表。
            ground_truths: ground truth 列表。

        Returns:
            最终答案列表。
        """
        results = []
        # 准备参数列表

        for query, ground_truth in zip(queries, ground_truths):
            result = self.run(query, ground_truth, run_config=run_config, topk=topk)
            results.append(result)
        return results

    def batch_predict(self, queries: List[str], ground_truths: List[str], run_config:RunnableConfig=RunnableConfig(llm={}), topk=1) -> List[str]:
        """
        执行批量的 Pipeline 流程：
        1. 使用 AgentTop 进行任务分解和决策。
        2. 使用 AgentLow 进行检索和总结。
        3. 生成最终答案。

        Args:
            queries: 用户查询文本列表。
            ground_truths: ground truth 列表。

        Returns:
            最终答案列表。
        """
        results = []
        # 准备参数列表

        for query, ground_truth in zip(queries, ground_truths):
            result = self.predict(query, ground_truth, run_config=run_config, topk=topk)
            results.append(result)
        return results

    # async def run(self, query: str, ground_truth: str, run_config: RunnableConfig = RunnableConfig(llm={}), topk: int = 1):
    #     query = query[-10000:]
        
    #     def _sync():
    #         return self.agent_low.sample_path(query, ground_truth, run_config=run_config, topk=topk)
    
    #     loop = asyncio.get_event_loop()
    #     golden_paths, negative_paths, query, groundtruth = await loop.run_in_executor(None, _sync)
    #     return golden_paths, negative_paths, query, groundtruth
    
    
    # async def predict(self, query: str, ground_truth: str, run_config: RunnableConfig = RunnableConfig(llm={}), topk: int = 1):
    #     query = query[-10000:]
    
    #     def _sync():
    #         return self.agent_low.predict(query, run_config=run_config, topk=topk)
    
    #     loop = asyncio.get_event_loop()
    #     final_answer, golden_paths, reverse_negative_paths = await loop.run_in_executor(None, _sync)
    #     return query, ground_truth, final_answer, golden_paths, reverse_negative_paths




    def run(self, query: str, ground_truth: str, run_config:RunnableConfig=RunnableConfig(llm={}), topk=1) -> str:
        """
        执行完整的 Pipeline 流程：
        1. 使用 AgentTop 进行任务分解和决策。
        2. 使用 AgentLow 进行检索和总结。
        3. 生成最终答案。

        Args:
            query: 用户查询文本。

        Returns:
            最终答案。
        """
        # 使用 AgentTop 处理查询
        query  = query[-10000:]

        golden_paths, negative_paths, query, groundtruth = self.agent_top.sample_path(query, ground_truth, run_config=run_config, topk=topk)
        return golden_paths, negative_paths, query, groundtruth


    
    def predict(self, query: str, ground_truth: str, run_config:RunnableConfig=RunnableConfig(llm={}), topk=1) -> str:
        """
        执行完整的 Pipeline 流程：
        1. 使用 AgentTop 进行任务分解和决策。
        2. 使用 AgentLow 进行检索和总结。
        3. 生成最终答案。

        Args:
            query: 用户查询文本。

        Returns:
            最终答案。
        """
        # 使用 AgentTop 处理查询
        query = query[-10000:] # 最好不要超过30000，不然可以学习kimi-rtinesearch的方式。

        final_answer, golden_paths, reverse_negative_paths  = self.agent_top.predict(query, run_config=run_config, topk=topk)
        return query, ground_truth, final_answer, golden_paths, reverse_negative_paths




    def case(self, query: str, run_config:RunnableConfig=RunnableConfig(llm={}), topk=1):
        final_answer, golden_paths, reverse_negative_paths  = self.agent_top.predict(query, run_config=run_config, topk=topk)
        return final_answer, golden_paths




@ray.remote(num_gpus=1,num_cpus=2) #和 create_sft_data_flask一起使用, 每个actor使用一个gpu; 不能占用过多GPU
class PureLLM:
    def __init__(self, llm: Any, embedding_model: Any, metapaths: List[Dict[str, str]], topk: int=3,  config=None, eval_path=None):
        """
        初始化 Pipeline，包含 AgentTop 和 AgentLow。

        Args:
            llm: 语言模型实例（用于 AgentTop 和 AgentLow）。
            embedding_model: 文本向量化模型（用于 AgentLow）。
            metapaths: 元路径列表，包含 name 和 description。
            ratio: 选择 meta-path 的比例（默认 0.3）。
        """
        # 初始化 AgentLow，传入 LLM 和 embedding_model
        self.device = torch.cuda.current_device() # 这样记录device也不知道行不行xs
        self.metapaths = metapaths
        self.llm, self.embedding_model = self.flask_llm_emb(config, eval_path) # 和flask一起使用


    def flask_llm_emb(self,config, eval_path=None):
        if eval_path is None:
            llm = get_llm_model(config, config['T_LLM'], api_base=None, path=config['T_LLM_PATH'],device=self.device)
        else:
            llm = get_llm_model(config, config['T_LLM'], api_base=None, path=eval_path, device=self.device)
        emb_model = get_emb_model(config, device=self.device)
        return llm, emb_model




    def batch_run(self, queries: List[str], ground_truths: List[str], run_config:RunnableConfig=RunnableConfig(llm={}), topk=1) -> List[str]:
        """
        执行批量的 Pipeline 流程：
        1. 使用 AgentTop 进行任务分解和决策。
        2. 使用 AgentLow 进行检索和总结。
        3. 生成最终答案。

        Args:
            queries: 用户查询文本列表。
            ground_truths: ground truth 列表。

        Returns:
            最终答案列表。
        """
        results = []
        # 准备参数列表

        for query, ground_truth in zip(queries, ground_truths):
            result = self.run(query, ground_truth, run_config=run_config, topk=topk)
            results.append(result)
        return results

    def batch_predict(self, queries: List[str], ground_truths: List[str], run_config:RunnableConfig=RunnableConfig(llm={}), topk=1) -> List[str]:
        """
        执行批量的 Pipeline 流程：
        1. 使用 AgentTop 进行任务分解和决策。
        2. 使用 AgentLow 进行检索和总结。
        3. 生成最终答案。

        Args:
            queries: 用户查询文本列表。
            ground_truths: ground truth 列表。

        Returns:
            最终答案列表。
        """
        results = []
        # 准备参数列表

        for query, ground_truth in zip(queries, ground_truths):
            result = self.predict(query, ground_truth, run_config=run_config, topk=topk)
            results.append(result)
        return results

        
    
    # async def run(self, run_config: RunnableConfig = RunnableConfig(llm={}), topk: int = 1):
    #     print("No custom run for these baselines")
    #     return None
        
    # async def predict(
    #     self,
    #     query: str,
    #     ground_truth: str,
    #     run_config: RunnableConfig = RunnableConfig(llm={}),
    #     topk: int = 1,
    # ) -> Tuple[str, str, str]:
    #     from config import config
    #     loop = asyncio.get_event_loop()
    
    #     if config['MODEL'] == 'meditron-7B':
    #         query = query[-2047:]
    #     else:
    #         query = query[-10000:]
    
    #     final_answer = await loop.run_in_executor(
    #         None,
    #         self.llm.invoke,
    #         query,
    #         run_config
    #     )
    
    #     return query, final_answer, ground_truth



    def run(self, run_config:RunnableConfig=RunnableConfig(llm={}), topk=1):
        print("No custom run for these baselines")

    def predict(self, query: str, ground_truth: str, run_config, topk=1) -> str:
        """
        执行完整的 Pipeline 流程：
        1. 使用 AgentTop 进行任务分解和决策。
        2. 使用 AgentLow 进行检索和总结。
        3. 生成最终答案。

        Args:
            query: 用户查询文本。

        Returns:
            最终答案。
        """
        from config import config

        if config['MODEL'] =='meditron-7B':
            query = query[-2047:] # 不要超过, 垃圾
        # if config['DATASET'] == 'eICU':
        query = query[-10000:] # 最好不要超过30000，不然可以学习kimi-rtinesearch的方式。
        final_answer = self.llm.invoke(query, config=run_config) # 必须要设定为invoke对象才能用
        # print("AAAAAAA", final_answer)

        return query, final_answer, ground_truth



####### 直接进行inference。这里非常必要，源自部分算法直接进行inference

def pure_llm(model_name, config, sft_mode=False):
    # vllm不支持chain输入
    hub_path = '/hpc2hdd/home/xxxs349/xxxc/huggingface/hub/'
    model_path = os.path.join(hub_path, model_name)
    # if vllm:
    #     llm = LLM(model_path)
    # else:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained( # 方便执行命令的时候使用 CUDA_VISIBLE_DEVICES=1,2,3 作为前缀
        model_path,
        device_map="auto",          # 自动分配GPU
        torch_dtype=torch.float16,  # 半精度节省显存
        quantization_config=config['quantization']
    )

    tokenizer.pad_token_id = tokenizer.eos_token_id  # 去除warning

    # 如果启用SFT模式，添加PEFT配置
    if sft_mode:
        peft_config = LoraConfig(
            r=config.get("lora_r", 8),
            lora_alpha=config.get("lora_alpha", 16),
            target_modules=config.get("target_modules", ["q_proj", "v_proj"]),
            lora_dropout=config.get("lora_dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()  # 打印可训练参数

        return model, tokenizer
    else:
        # 创建 HuggingFace Pipeline
        hf_pipeline = Pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            # device=1,#int(config['GPU']) if torch.cuda.is_available() and config['USE_CUDA'] else "cpu",
            # 如果有GPU，可以使用 device=0
            return_full_text=False,
            **config.get("generation", {})  # 自动注入配置
        )
        llm = HuggingFacePipeline(pipeline=hf_pipeline)
        return llm




def special_pro(raw_out):
    if isinstance(raw_out, list):
        return [f"Answer: {item}" for item in raw_out]
    elif isinstance(raw_out, str):
        return f"Answer: {raw_out}"
    else:
        return raw_out  # 如果不是列表或字符串，返回原值

def pure_llm_response(llm, input_variables, template, special_func=None):
    """
    input_variables dict
    """
    # 构建 prompt
    prompt = PromptTemplate(
        input_variables=list(input_variables.keys()),
        template=template)
    # 调用语言模型生成最终答案（这里假设使用 OpenAI GPT）
    # 调用外部传入的 LLM 生成总结
    chain = prompt | llm
    answer = chain.invoke(input_variables)
    if special_func is not None:
        answer = special_func(answer)
    return answer


def pure_llm_sft(model, tokenizer, input_variables, template,
                      temperature=0.7, max_new_tokens=200, **kwargs):
    """
    更新后的推理函数
    """
    # 构建prompt
    prompt = PromptTemplate(
        input_variables=list(input_variables.keys()),
        template=template
    ).format(**input_variables)

    # 生成参数
    generation_config = {
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id
    }

    # 生成文本
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = llm.generate(
        **inputs,
        **generation_config
    )

    # 解码结果
    answer = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    return answer
def pure_llm_batch(llm, input_variables, template, special_func=None):
    """
    input_variables [dict]
    """
    # 构建 prompt
    prompt = PromptTemplate(
        input_variables=list(input_variables[0].keys()),
        template=template)
    chain = prompt | llm
    answers = chain.batch(input_variables)
    if special_func is not None:
        answers = special_func(answers)
    return answers


def pure_llm_batch_sft(
        llm,
        tokenizer,
        batch_input_variables,  # 改为接收多个输入的列表
        template,
        temperature=0.7,
        max_new_tokens=200,
        batch_size=4,  # 新增批量大小控制
        **kwargs
):
    """
    支持批量输入的推理函数
    batch_input_variables: List[dict] 输入字典的列表
    """
    # 批量构建prompt
    prompts = [
        PromptTemplate(
            input_variables=list(inputs.keys()),
            template=template
        ).format(**inputs)
        for inputs in batch_input_variables
    ]

    # 批量编码（自动填充padding）
    encoded_inputs: BatchEncoding = tokenizer(
        prompts,
        padding=True,  # 启用填充
        truncation=True,  # 启用截断
        max_length=1024,  # 设置最大输入长度
        return_tensors="pt"
    ).to(llm.device)  # 自动分配到模型所在设备

    # 生成参数配置
    generation_config = {
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id
    }

    # 批量生成
    outputs = llm.generate(
        **encoded_inputs,
        **generation_config
    )

    # 批量解码（跳过特殊token和输入部分）
    answers = []
    for i, output in enumerate(outputs):
        input_length = encoded_inputs.input_ids[i].shape[0]
        generated_tokens = output[input_length:]  # 只保留生成部分
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        answers.append(answer)

    return answers

if __name__ == '__main__': # 这些放入main函数中
    # 测试pure llm
    from instructions_template import task_templates
    template = task_templates['MOR']

    config = {
        'GPU': 0,
        'USE_CUDA':True,
        "generation": {
            "temperature": 1, # higher=random
            "max_new_tokens": 300, # new token
            "do_sample": True,
            "top_p": 0.85 # keep token 》prob
        },
        'quantization':BitsAndBytesConfig(load_in_8bit=True),

    }
    model_name = 'qwen25-32B'#'qwen25-7B'#'meditron-7b' # llama3-7B
    llm = pure_llm(model_name, config) # vllm适合起服务
    input_variables = [
        {
            "disease_info": [
                "Diabetes Mellitus",
                "Hypertension",
                "Coronary Artery Disease",
                "Asthma",
                "Chronic Obstructive Pulmonary Disease (COPD)"
            ],
            "procedure_info": [
                "Metformin",
                "Lisinopril",
                "Atorvastatin",
                "Salbutamol Inhaler",
                "Fluticasone Propionate/Salmeterol"
            ],
            "prescription_info": [
                "Metformin",
                "Lisinopril",
                "Atorvastatin",
                "Salbutamol Inhaler",
                "Fluticasone Propionate/Salmeterol"
            ]
        },
        {
            "disease_info": [
                "Hyperlipidemia",
                "Osteoarthritis",
                "Depression",
                "Thyroiditis",
                "Gastroesophageal Reflux Disease (GERD)"
            ],
            "procedure_info": [
                "Hyperlipidemia",
                "Osteoarthritis",
                "Depression",
                "Thyroiditis",
                "Gastroesophageal Reflux Disease (GERD)"
            ],
            "prescription_info": [
                "Simvastatin",
                "Ibuprofen",
                "Fluoxetine",
                "Levothyroxine",
                "Omeprazole"
            ]
        }
    ]
    answer = pure_llm_response(llm, input_variables[0], template, special_func=special_pro)

    answer = pure_llm_batch(llm, input_variables, template, special_func=special_pro)
