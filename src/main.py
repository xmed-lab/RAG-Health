"""
# File       : dataset.py
# Time       ：26/3/2025 9:47 am
# Author     ：Any
# version    ：python
# Description：
"""


import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import subprocess
import gc
import time
import torch
from config import config
from dataset import get_task_fn, convert_dataset, create_sft_data_think_flask, re_construct_format, get_map_system, create_sft_data_flask, re_construct_format_think, create_sft_data_infer_flask
from dataset import re_construct_format_infer, re_construct_format_flash, create_sft_data_flash_flask
from pyhealth.datasets import MIMIC3Dataset, MIMIC4Dataset, eICUDataset, OMOPDataset # SampleEHRDataset
from pyhealth.datasets import SampleEHRDataset as SampleEHRDatasets
from pic_parse import PICDataset
from omix_parse import OMIXDataset
from shot_parse import SHOTDataset
from bhc_parse import BHCDataset, preprocess_bhc
from utils import split_by_patient, load_pickle, save_pickle, set_random_seed, copy_file
from dataloader import get_dataloader, get_special_input
import itertools
from evaluates import evaluate_generation_flask, evaluate_generation_think_flask, evaluate_generation_infer_flask
from utils import generate_rare_disease, generate_rare_patient

set_random_seed(config['SEED'])


class SampleEHRDataset(SampleEHRDatasets):
    # 在制造数据时候可以避免繁琐的检查,
    def _validate(self):
        return True

def run_single_config(config, exp_num='3'):
    # 1. 读取数据
    root_to = '/hpc2hdd/home/xxxs349/xxxc/RAGHealth/data/{}/{}/processed/'.format(config['TASK'], config['DATASET'])
    if not os.path.exists(root_to):
        # 如果不存在，则创建路径
        os.makedirs(root_to)

    task_fn, mode, task_mode = get_task_fn(config)
    if not os.path.exists(root_to + 'datasets_pre_stand.pkl'):
        print("Prepare dataset!")
        if config['DATASET'] == 'MIII':
            base_dataset = MIMIC3Dataset(
                root="/home/xxxc/HyperHealth/data/physionet.org/files/mimiciii/1.4",
                tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
                code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": config['ATCLEVEL']}})}, # 这里graphcare的ATC-level是3；和我们在data阶段有差别
                dev=False,
                refresh_cache=False,
            )
        elif config['DATASET'] == 'eICU':
            base_dataset = eICUDataset(
                root="/home/xxxc/HyperHealth/data/physionet.org/files/eicu-crd/2.0",
                tables=["diagnosis", "medication", "physicalExam", "treatment", "admissionDx"],
                dev=False,
                refresh_cache=False,
            )
        elif config['DATASET'] == 'MIV':
            base_dataset = MIMIC4Dataset(
                root="/home/xxxc/HyperHealth/data/physionet.org/files/mimiciv/2.0/hosp",
                tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
                code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": config['ATCLEVEL']}})}, # 4
                dev=False,
                refresh_cache=False,
            )
        elif config['DATASET'] == 'MIV-Note': # 这里可以通过subject_id进行联系。因为有一部分数据被MIMIC官方雪藏了。所有用2.2以上的版本
            base_dataset = MIMIC4Dataset(
                root="/home/xxxc/HyperHealth/data/physionet.org/files/mimiciv/2.0/hosp",
                tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
                code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": config['ATCLEVEL']}})}, # 4
                dev=False,
                refresh_cache=False,
            )
        elif config['DATASET'] == 'PIC':
            base_dataset = PICDataset(
                root="/home/xxxc/HyperHealth/data/physionet.org/files/picdb/1.1.0/",
                tables=["DIAGNOSES_ICD","PRESCRIPTIONS","LABEVENTS"],
                dev=False,
                refresh_cache=False,
            )
        elif config['DATASET'] == "EHR-SHOT":
            base_dataset = SHOTDataset(
                root="/home/xxxc/HyperHealth/data/EHRSHOT_ASSETS/data/",
                tables=["ehrshot"],
                dev=False,
                refresh_cache=False,
            )
        elif config['DATASET'] == 'OMOP':
            base_dataset = OMOPDataset(
                root="https://storage.googleapis.com/pyhealth/synpuf1k_omop_cdm_5.2.2",
                tables=["condition_occurrence",
                        "procedure_occurrence",
                        "drug_exposure",
                        "measurement"],
                dev=False,
                refresh_cache=False,
            )
        elif config['DATASET'] == 'OMIX':
            base_dataset = OMIXDataset(
                root="/home/xxxc/HyperHealth/data/DataTable2/",
                tables=[
                    "Diagnosis",
                    "Lab",
                    "Medication",
                ],
                dev=False,
                refresh_cache=True,
            )
        else:
            print("No such dataset!")
            return

        base_dataset.stat()
        # set task
        sample_dataset = base_dataset.set_task(task_fn)
        sample_dataset.stat()
        samples = sample_dataset.samples
        save_pickle(samples, root_to + 'datasets_pre_stand.pkl')

        print("initial dataset done!")
        print("Please run again!")
        return
    else:
        start = time.time()
        samples = load_pickle(root_to + 'datasets_pre_stand.pkl')
        # for group analysis
        disease_group = generate_rare_disease(samples, config['RARE_THRES'], root_to, task=config['TASK'])
        generate_rare_patient(samples, disease_group['group_disease'], root_to)

        # # group
        # generate_patient_group(samples, root_to)
        # p_grouped = load_pickle(root_to + 'group_patient.pkl')

        end = time.time()
        print("Load data done! Cost time {} s".format(end-start))


        # 这里会花大量的时间, DEV 模式在这里不需要。可以通过Step 3的循环来减少数据
        try:
            reserved_tensor = 0#torch.ones((10, 10)).to('cuda:' + config['GPU']) # 占据GPU
            print("GPU Memory Usage", torch.cuda.memory_allocated('cuda:' + config['GPU']) / 1024 / 1024 / 1024,
                  "GB")
            if os.path.exists(root_to + 'train_dataset.pkl'):
                print("Load train dataset from cache!")
                sample_dataset = convert_dataset(samples, dataset_name=config['DATASET'], task_name=config['TASK'], valid=False, all=True) # 需要查看是否需要load_code_convert
                # warm_test, cold_test = get_warm_cold_split(samples)
                train_samples = load_pickle(root_to + 'train_dataset.pkl')
                val_samples = load_pickle(root_to + 'val_dataset.pkl')
                test_samples = load_pickle(root_to + 'test_dataset.pkl')
                train_dataset = convert_dataset(train_samples, dataset_name=config['DATASET'], task_name=config['TASK'], valid=False)
                val_dataset = convert_dataset(val_samples, dataset_name=config['DATASET'], task_name=config['TASK'], valid=False)
                test_dataset = convert_dataset(test_samples, dataset_name=config['DATASET'], task_name=config['TASK'], valid=False)
            else:
                print("Create train dataset!")
                sample_dataset = convert_dataset(samples, dataset_name=config['DATASET'], task_name=config['TASK'], valid=True) # 需要查看是否需要load_code_convert
                # warm_test, cold_test = get_warm_cold_split(samples)
                train_dataset, val_dataset, test_dataset = split_by_patient(
                    sample_dataset, [config['RATIO'], (1 - config['RATIO']) / 2, (1 - config['RATIO']) / 2],
                    train_ratio=1.0,  # Train test split
                    warm_cold=False, # 所有的warm一次性看完。
                    seed=config['SEED']
                )  # 这样似乎更快，固定随机种子的时候是一样的；
                # train_samples, _, test_samples = achieve_samples(train_dataset), _, achieve_samples(test_dataset)
                save_pickle(train_dataset, root_to + 'train_dataset.pkl')
                save_pickle(val_dataset, root_to + 'val_dataset.pkl')
                save_pickle(test_dataset, root_to + 'test_dataset.pkl')
        finally:
            reserved_tensor =0
            # torch.cuda.empty_cache() # 会突然爆oom

        endt = time.time()
        print('Train Dataset done!, Cost {} s'.format(endt - end))

    # STEP 2: load dataloader
    collate_fn = get_special_input(config)
    train_dataloader = get_dataloader(train_dataset, batch_size=config['BATCH'], shuffle=True, drop_last=True, collate_fn=collate_fn)
    # val_dataloader = get_dataloader(val_dataset, batch_size=config['BATCH']*5, shuffle=False, drop_last=True, collate_fn=collate_fn)
    test_dataloader = get_dataloader(test_dataset, batch_size=config['BATCH'], shuffle=False, drop_last=False, collate_fn=collate_fn)
    load_dataloader = time.time()
    print('Dataloader done!, Cost {} s'.format(load_dataloader - endt))

    # cache clear
    # torch.cuda.empty_cache() # 占用内存清理
    gc.collect()

    # STEP 3: create sft data
    map_dict = get_map_system(config) # ID-> Text

    #create_sft_data_flask(train_dataloader, root_to + 'train_sft_data.jsonl', config, map_dict, task_mode=task_mode, train_mode=True) # trans应该是全的。有完整的QA
    # create_sft_data(val_dataloader, root_to + 'val_sft_data.pkl', config, map_dict, task_mode=task_mode, train_mode=False)
   # create_sft_data_flask(test_dataloader, root_to + 'test_sft_data.jsonl', config, map_dict, task_mode=task_mode, train_mode=False) # 注意，下面test中仅用pure，以便在inference进行
    # 
    # # Step 4: 按LLM Factory格式构建数据
    meta_path_dic = load_pickle('/hpc2hdd/home/xxxs349/xxxc/RAGHealth/data/ready/meta_path.pkl') # 会报错
    meta_path_dic = {key: value for key, value in meta_path_dic.items() if key in map(str, range(0, 10))} # 和create_sft中保持一致。

    
    # 
    re_construct_format(root_to + 'train_sft_data.jsonl',root_to,meta_path_dic, train_mode=True) # mixed json. all data没有经过过滤。
    # re_construct_format(root_to + 'val_sft_data.jsonl',root_to,meta_path_dic, train_mode=False)
    re_construct_format(root_to + 'test_sft_data.jsonl',root_to,meta_path_dic, train_mode=False)

    # Step 5: 迁移数据到LLAMA Factory
    copy_file(src=root_to + 'all_data.jsonl', dst='all_data_{}_{}.jsonl'.format(config['DATASET'],config['TASK'])) # for sft; all data 是reformat golden path
    copy_file(src=root_to + 'train_sft_data.jsonl', dst='train_sft_data_{}_{}.jsonl'.format(config['DATASET'],config['TASK'])) # 供查找golden_path/negative path
    copy_file(src=root_to + 'test_sft_data.jsonl'.format(config['MODEL']), dst='test_sft_data_{}_{}.jsonl'.format(config['DATASET'], config[
        'TASK']), test_mode=True)  # 供查找golden_path/negative path
    copy_file(src=root_to + 'pure_path.jsonl', dst='pure_path_{}_{}.jsonl'.format(config['DATASET'],config['TASK'])) #  只有QA pair。
    print("All data is ready! Please check the files in the LLAMA Factory directory.")
 


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Hi! RAGHealth!')
    print('You are runnning model {}, on dataset {}, task {}.'.format(
        config['MODEL'], config['DATASET'], config['TASK']))
    task_fn, mode, task_mode = get_task_fn(config)

    train = False # 必须先train保证数据处理，模型训练完成, 其实直接inference用的数据都是一套
    root_to = '/hpc2hdd/home/xxxs349/xxxc/RAGHealth/data/{}/{}/processed/'.format(config['TASK'], config['DATASET'])
    # 采集数据 & 训练模型
    if train:
        if config['MODEL'] == 'ours':
            run_single_config(config)
   
    else:
        model_path = config['MODEL_PATH']
        if config['MODEL']== 'ours':
            test_data_path = root_to + 'test_sft_data.jsonl'
            output_path = root_to + 'test_output.jsonl'
            print("We need to evaluate the model at: {}".format(model_path)) # 这个应该和merge_lora里面的一摸一样。
            assert os.path.exists(model_path), "Model path does not exist!"
            p_grouped = load_pickle(root_to + 'rare_patient.pkl')
            
            test_samples = load_pickle(root_to + 'test_dataset.pkl')
            patient_ids = [sample['patient_id'] for sample in test_samples]
            evaluate_generation_flask(model_path, test_data_path, config, output_path, task_mode=task_mode,run_config=config['run_config'], topk=config['TOPK'], p_grouped=[patient_ids,p_grouped])
