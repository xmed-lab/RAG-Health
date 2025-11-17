# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : bhc_parse.py
# Time       ：20/5/2025 2:05 pm
# Author     ：Any
# version    ：python 
# Description：summary 读取bhc文件, 本质是个summarY
"""
import os
import numpy as np
import pandas as pd
class BHCDataset(object):
    """
    BHC数据集
    """

    def __init__(self, root, table):
        """
        :param root: 数据集路径
        :param tables: 数据集表格
        """
        self.root = root
        self.table = table


    def get_data(self):
        """
        获取数据集
        :return: 数据集
        """
        table_path = os.path.join(self.root, self.table) + '.csv'
        if not os.path.exists(table_path):
            raise FileNotFoundError(f"Table {self.table} not found in {self.root}")
        else:
            data = pd.read_csv(table_path)
        return data



def preprocess_bhc(base_dataset, filter_condition=None):
    """
    预处理数据集
    :param base_dataset: 数据集
    :return: 预处理后的数据集
    """
    # 这里可以添加一些数据预处理的代码
    # 例如去除缺失值、标准化等
    # 这里以去除缺失值为例
    df = base_dataset.dropna()

    if filter_condition is not None:
        filtered_df = df[(df['input_tokens'] >= filter_condition[0]) & (df['target_tokens'] >= filter_condition[1])]
    else:
        filtered_df = df

    # 转为json
    # 存储为多个 JSON
    json_list = []
    for _, row in filtered_df.iterrows():
        json_entry = {
            "instruction": "",
            "input": row['input'],
            "output": row['target']
        }
        json_list.append(json_entry)
    json_list = json_list[:1000] # 不然太多了
    return json_list


# def split_data_bhc(json_list, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=None):
#     # 确保比例和为1
#     assert train_ratio + val_ratio + test_ratio == 1, "比例必须和为1"
#     if seed is not None:
#         np.random.seed(seed)
#
#     # 打乱数据
#     np.random.shuffle(json_list)
#
#     total_size = len(json_list)
#     train_size = int(total_size * train_ratio)
#     val_size = int(total_size * val_ratio)
#
#     train_data = json_list[:train_size]
#     val_data = json_list[train_size:train_size + val_size]
#     test_data = json_list[train_size + val_size:]
#
#     return train_data, val_data, test_data


def split_data_bhc(json_list, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=None):
    # 确保比例和为1
    assert train_ratio + val_ratio + test_ratio == 1, "比例必须和为1"
    if seed is not None:
        np.random.seed(seed)

    # 打乱数据
    np.random.shuffle(json_list)

    total_size = len(json_list)
    train_size = int(total_size * 0.5)
    val_size = int(total_size * val_ratio)

    train_data = json_list[:train_size]
    val_data = [] # json_list[train_size:train_size + val_size]
    test_data = json_list[train_size:]

    return train_data, val_data, test_data
