# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : config.py
# Time       ：5/3/2025 9:48 am
# Author     ：Any
# version    ：python 
# Description：整体train/inference的config文件
"""
from langchain.schema.runnable import RunnableConfig

MIII_PARAMS = {
    'FEATURE' : ['conditions', 'procedures', 'drugs'],  # drug_HIST两层含义, 只让他看conditions和procedures
    'SEED': 528,
    'USE_CUDA': True,
    'GPU': '2',
    'EPOCH': 50,
    'DIM': 128,
    'HIDDEN': 128,
    'LR': 1e-3,
    'BATCH': 24, # 32 条铁定会爆炸
    'DROPOUT': 0.1,
    'WD': 0., # 1e-3,
    'RNN_LAYERS': 2,
    'MULTI': 0,
    'MAXSEQ': 10,
    'MAXCODESEQ': 512,
}

BHC_PARAMS = {
    'FEATURE' : ['conditions', 'procedures', 'drugs'],  # drug_HIST两层含义, 只让他看conditions和procedures
    'SEED': 528,
    'USE_CUDA': True,
    'GPU': '2',
    'EPOCH': 50,
    'DIM': 128,
    'HIDDEN': 128,
    'LR': 1e-3,
    'BATCH': 4, # 32 条铁定会爆炸
    'DROPOUT': 0.1,
    'WD': 0.,#1e-3,
    'RNN_LAYERS': 2,
    'MULTI': 0,
    'MAXSEQ': 10,
    'MAXCODESEQ': 512,
}

MEDQA_PARAMS = {
    'FEATURE' : ['conditions', 'procedures', 'drugs'],  # drug_HIST两层含义, 只让他看conditions和procedures
    'SEED': 528,
    'USE_CUDA': True,
    'GPU': '2',
    'EPOCH': 50,
    'DIM': 128,
    'HIDDEN': 128,
    'LR': 1e-3,
    'BATCH': 24, # 32 条铁定会爆炸
    'DROPOUT': 0.1,
    'WD': 0.,#1e-3,
    'RNN_LAYERS': 2,
    'MULTI': 0,
    'MAXSEQ': 10,
    'MAXCODESEQ': 512,
}

eICU_PARAMS = {
    'FEATURE': ['conditions', 'procedures', 'drugs'],  # drug_HIST两层含义, 只让他看conditions和procedures
    'SEED': 528,
    'USE_CUDA': True,
    'GPU': '2',
    'EPOCH': 50,
    'DIM': 128,
    'HIDDEN': 128,
    'LR': 1e-3, #
    'BATCH': 32, # CORGAN要变小的
    'DROPOUT': 0.3, # LOS考虑0.5 PHE考虑0.3
    'WD': 5e-4,# PHE 5e-4 LOS 0
    'RNN_LAYERS': 2,
    'MULTI': 0,
    'MAXSEQ': 10, # 太长会有问题
    'MAXCODESEQ': 512,
}

MIV_PARAMS = {
    'FEATURE' : ['conditions', 'procedures', 'drugs'],
    'SEED': 528,
    'USE_CUDA': True,
    'GPU': '0',
    'EPOCH': 50,
    'DIM': 128,
    'HIDDEN': 128,
    'LR': 1e-3,
    'BATCH': 32,
    'DROPOUT': 0.3, # 0. # PHE
    'WD': 5e-4, #0.
    'RNN_LAYERS': 2,
    'MULTI': 0,
    'MAXSEQ': 10,
    'MAXCODESEQ': 512,
}

EHR_PARAMS = {
    'FEATURE': ['conditions', 'procedures', 'drugs'],  # drug_HIST两层含义, 只让他看conditions和procedures
    'SEED': 528,
    'USE_CUDA': True,
    'GPU': '3',
    'EPOCH': 50,
    'DIM': 128,
    'HIDDEN': 128,
    'LR': 1e-3,
    'BATCH': 8,
    'DROPOUT': 0.3,
    'WD': 5e-4,
    'RNN_LAYERS': 2,
    'MULTI': 0,
    'MAXSEQ': 10,
    'MAXCODESEQ': 512,
}

OMOP_PARAMS = {
    'FEATURE': ['conditions', 'procedures', 'drugs'],  # drug_HIST两层含义, 只让他看conditions和procedures
    'SEED': 528,
    'USE_CUDA': True,
    'GPU': '5',
    'EPOCH': 50,
    'DIM': 128,
    'HIDDEN': 128,
    'LR': 1e-3,
    'BATCH': 32,
    'DROPOUT': 0.5,
    'WD': 1e-2, # bu
    'RNN_LAYERS': 2,
    'MULTI': 0,
    'MAXSEQ': 10,
    'MAXCODESEQ': 512,
}

class UNIFYCONFIG(): # 不要有drugs
    """DRL config"""
    # data_info parameter
    DEV = False # 是否使用sample子集
    MODEL = 'KARE'# 'ours'#'Search-o1'#'meditron-7B'#'MedRAG' #"ours" # ours, CoT,LightRAG, KARE, MedRAG
    PLM = "Clinical-BERT" # ours # flexcare los 增大学习率；prism phe增大; Clinical-BERT; Sap-BERT; BioGPT
    TASK = 'LOS'#'SINGLE'#'SUMMARY'#'MOR'
    DATASET = 'MIV'#'MedQA'#'MIV-Note-BHC'#'MIII'
    LABEL = 'labels'


    ATCLEVEL = 3
    RATIO = 0.6 # train-test split
    THRES = 0.4 # pred threshold
    RARE_THRES = 0.8 # for group anaylss

    LLM = 'qwen25-7B'#'qwen25-7B' # VLLM挂载. 如果使用本地模式，则不能进行demo阶段的temperature调整。
    LLM_PATH = "/nfs/scratch/xxxc/huggingface/hub/qwen25-7B"
    T_LLM = 'qwen25-7B'#'qwen-plus'#'deepseek-v3'#'LLAMA3-8B' # Teacher LLM是更强的LLMs; 统一调用API; 'LLAMA3-3B'#除了qwen都要改
    T_LLM_PATH =  "/nfs/scratch/xxxc/huggingface/hub/qwen25-7B" #''#'/home/xxxc/huggingface/hub/llama3-8B/' # 注意最后一个下划线。vllm server挂载和python 不一致
    EMB = 'E5' # 一般不要换，不然得重新构建embedding。
    EMB_PATH = "/nfs/scratch/xxxc/huggingface/hub/e5-v2"
    KG_PATH = "/home/xxxc/RAGHealth/data/raw/primekg/"
    TEXT_PATH = "/home/xxxc/RAGHealth/data/ready/pubmed_results_node_20250314_112754.json"
    META_RATIO = 0.1 # meta-path
    TOPK = 1 # entity
    DEPTH = 5
    MODEL_PATH = "/nfs/scratch/xxxc/huggingface/hub/qwen25-7B" #'/home/xxxc/RAGHealth/LLaMA-Factory/output/llama3_lora_reward' # final model path； 需要最后统一下名字
    # statistic
    run_config_dict = {"temperature": 0.1, "max_tokens": 256, "max_new_tokens":256, "top_p": 0.9, "top_k": 50} # for api/vllm
    # dynamic
    run_config = RunnableConfig(configurable={"temperature": 0.1, "max_tokens": 256, "max_new_tokens":256, "top_p": 0.9, "top_k": 50})
    # train parameter
    DATASET_PARAMS = {
        'MIII': MIII_PARAMS,
        'MIV': MIV_PARAMS,
        'eICU': eICU_PARAMS,
        'EHR-SHOT': EHR_PARAMS,
        'OMOP': OMOP_PARAMS,
        'MIV-Note-BHC': BHC_PARAMS, # 这里的MIV-Note-BHC是为了和MIII区分开来，实际上是一样的
        'MedQA': MEDQA_PARAMS, # 这里的MedQA是为了和MIII区分开来，实际上是一样的
    }

    @classmethod
    def get_params(cls):
        return cls.DATASET_PARAMS.get(cls.DATASET, {})

    # log
    LOGDIR = '/home/xxxc/RAGHealth/log/ckpt/'

config = {**vars(UNIFYCONFIG), **UNIFYCONFIG.get_params()}
config = {k: v for k, v in config.items() if not k.startswith('__')}
