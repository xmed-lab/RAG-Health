# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : dataloader.py
# Time       ：5/3/2025 9:47 am
# Author     ：Any
# version    ：python 
# Description：
"""
import torch
from torch.utils.data import DataLoader, Dataset

def collate_fn_dict(batch):
    return {key: [d[key] for d in batch] for key in batch[0]} # conditions: [B,V,M]

def get_dataloader(dataset, batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn_dict):
    """for first, third stage"""
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=drop_last
    )

    return dataloader


def collate_fn_note(batch):
    data = {key: [d[key] for d in batch] for key in batch[0]} # 干脆直接在论文里处理。就是model forward多一个部分
    # data['note'] = [[d['note']] * len(d['conditions']) for d in batch] # 把这个note复制到每个condition上，不然存储的太大了。
    return data # conditions: [B,V,M]


def get_special_input(config):
    """需要多样的数据输入，例如额外增加mask"""
    if config['MODEL']== 'PRISM':
        pass
    elif config['DATASET'] == 'MIV-Note':
        return collate_fn_note
    else:
        return collate_fn_dict
