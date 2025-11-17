# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : analysis.py
# Time       ：12/7/2025 10:50 pm
# Author     ：Any
# version    ：python 
# Description：#  用GPT生成一些必要的case DATA/descriptions，以供模仿
"""
import os
import pandas as pd
from utils import llm_generation
from instructions_template import sft_metapth_desc_instructions

def meta_path_desc():
    root_dir = "/home/xxxcbo/RAGHealth/data/raw/primekg"
    root_to_dir = "/home/xxxcbo/RAGHealth/data/ready"
    # 读取KG
    kg_path = os.path.join(root_dir, "kg.csv")
    data = pd.read_csv(kg_path)
    print(data.head())

    # 提取所有的类型关系
    unique_values = data[['x_type', 'relation', 'y_type']].drop_duplicates()
    result = unique_values.groupby(['x_type', 'y_type'])['relation'].unique().reset_index() # dis_relation有18种; relation有30种
    # tmp_metapath = [f"{row['A']}-to-{row['B']}" for index, row in unique_entity_values.iterrows()]
    result['coarse_path'] = result.apply(lambda x: f"{x['x_type']}-to-{x['y_type']}", axis=1)
    result['fine_path'] = result.apply(
        lambda x: [f"{x['x_type']}-{sub}-{x['y_type']}" for sub in x['relation']],
        axis=1
    )    # 存储到dict中

    # 生成一些meta-path的描述，其实就是coarse-grained metapath, or subdataset的描述
    result['generate_desc'] = result.apply(lambda x: llm_generation(result['fine_path']), axis = 1)

    data['fine_path'] =  data.apply(lambda x: f"{x['x_type']}-{x['relation']}-{x['y_type']}", axis=1)
    data['coarse_path'] = data.apply(lambda x: f"{x['x_type']}-to-{x['y_type']}", axis=1) # 可以直接构建图，然后进行meta_PATH采样

    # save
    meta_path_desc = os.path.join(root_to_dir, "meta_path_desc.csv")
    data_with_path = os.path.join(root_to_dir, "data_with_path.csv")
    result.to_csv(meta_path_desc, index=False)
    data.to_csv(data_with_path, index=False)
    print("ALL KG Process Done!")

def large_llm_example():
    pass
