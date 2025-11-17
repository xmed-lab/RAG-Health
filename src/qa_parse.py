# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : qa_parse.py
# Time       ：21/5/2025 2:48 pm
# Author     ：Any
# version    ：python 
# Description：一些常见的QA
"""
import os
import json
import numpy as np
import pandas as pd
class MedQADataset:

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
        table_path = os.path.join(self.root, self.table) + '.csv' # 感觉QA dataset可能需要shuffle，sample
        if not os.path.exists(table_path):
            raise FileNotFoundError(f"Table {self.table} not found in {self.root}")
        else:
            data = pd.read_csv(table_path)
        return data


class CommonQADataset:
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
        train_table_path = self.root + 'train_' + self.table + '.csv'  # 感觉QA dataset可能需要shuffle，sample
        test_table_path = self.root + 'test_' + self.table + '.csv'
        if not os.path.exists(train_table_path) or not os.path.exists(test_table_path):
            raise FileNotFoundError(f"Table {self.table} not found in {self.root}")
        else:
            train_data = pd.read_csv(train_table_path)
            test_data = pd.read_csv(test_table_path)

        print("Len train data:", len(train_data), "Len test data:", len(test_data))
        return [train_data, test_data]


def preprocess_medqa(base_dataset, filter_condition=None): # return formatedjson反正
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
            "input": row['question'] + row['options'],
            "output": row['answer']
        }
        json_list.append(json_entry)

    return json_list



def preprocess_common(base_dataset, filter_condition=None): # return formatedjson反正
    train_df, test_df = base_dataset

    if filter_condition is not None:
        filtered_train_df = train_df[(train_df['input_tokens'] >= filter_condition[0]) & (train_df['target_tokens'] >= filter_condition[1])]
        filtered_test_df = test_df[(test_df['input_tokens'] >= filter_condition[0]) & (test_df['target_tokens'] >= filter_condition[1])]
    else:
        filtered_train_df = train_df
        filtered_test_df = test_df


    # 转为json
    # 存储为多个 JSON
    json_list = []
    for _, row in filtered_train_df.iterrows():
        json_entry = {
            "instruction": "",
            "input": 'Query:\n' + row['question'] +'\nOptions:\n'+ row['answer_raw'] if len(row['context']) <2
            else 'Context:\n' + row['context']  +'Query:\n' + row['question'] +'\nOptions:\n'+ row['answer_raw'],
            "output": row['correct_answer']
        }
        json_list.append(json_entry)
        
    json_test_list = []
    for _, row in filtered_test_df.iterrows():
        json_entry = {
            "instruction": "",
            "input": 'Query:\n' + row['question'] +'\nOptions:\n'+ row['answer_raw'] if len(row['context']) <2
            else 'Context:\n' + row['context']  +'Query:\n' + row['question'] +'\nOptions:\n'+ row['answer_raw'],
            "output": row['correct_answer']
        }
        json_test_list.append(json_entry)

    return {'train':json_list, 'test':json_test_list}
def presplit_qa(path):
    # 先前已经搞定了的split
    with open(path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    json_list = []
    for row in data:
        json_entry = {
            "instruction": "",
            "input": row['question'] + "\n".join([f"{k}. {v}" for k, v in row['options'].items()]),
            "output": row['answer_idx']
        }
        json_list.append(json_entry)
    return json_list

def split_data_qa(json_list, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=None):
    # 确保比例和为1, 先前没搞定split的。
    assert train_ratio + val_ratio + test_ratio == 1, "比例必须和为1"
    if seed is not None:
        np.random.seed(seed)

    # 打乱数据
    np.random.shuffle(json_list)

    total_size = len(json_list)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    train_data = json_list[:train_size]
    val_data = json_list[train_size:train_size + val_size]
    test_data = json_list[train_size + val_size:]

    return train_data, val_data, test_data
