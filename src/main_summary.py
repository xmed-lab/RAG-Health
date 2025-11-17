import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import subprocess
import gc
import time
import torch
from config import config
from dataset import get_task_fn, convert_dataset, generate_patient_group, re_construct_format, \
    get_map_system, create_sft_data_flask, create_sft_data_think_flask, re_construct_format_think, create_sft_data_infer_flask, re_construct_format_infer, create_sft_data_flash_flask, re_construct_format_flash
from pyhealth.datasets import MIMIC3Dataset, MIMIC4Dataset, eICUDataset, OMOPDataset  # SampleEHRDataset
from pyhealth.datasets import SampleEHRDataset as SampleEHRDatasets
from pic_parse import PICDataset
from omix_parse import OMIXDataset
from shot_parse import SHOTDataset
from bhc_parse import BHCDataset, preprocess_bhc, split_data_bhc
from utils import split_by_patient, load_pickle, save_pickle, set_random_seed, copy_file
from dataloader import get_dataloader, get_special_input
from evaluates import evaluate_generation_flask, evaluate_generation_think_flask, evaluate_generation_infer_flask

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

        if config['DATASET'] == 'MIV-Note-BHC': # 感觉非选择题都可以放在这里
            base_dataset = BHCDataset(
                root="/hpc2hdd/home/xxxs349/xxxc/HyperHealth/data/physionet.org/files/labelled-notes-hospital-course/1.2.0",
                table='mimic-iv-bhc'
            )
            base_dataset = base_dataset.get_data()
            preprocess = preprocess_bhc
        else:
            print("No such dataset!")
            return

        samples = preprocess(base_dataset)
        save_pickle(samples, root_to + 'datasets_pre_stand.pkl')

        print("initial dataset done!")
        print("Please run again!")
        return
    else:
        start = time.time()
        samples = load_pickle(root_to + 'datasets_pre_stand.pkl')

        # # group
        # generate_patient_group(samples, root_to)
        # p_grouped = load_pickle(root_to + 'group_patient.pkl')

        end = time.time()
        print("Load data done! Cost time {} s".format(end - start))


        # 这里会花大量的时间
        try:
            reserved_tensor = torch.ones((10, 10)).to('cuda:' + config['GPU'])  # 占据GPU
            print("GPU Memory Usage", torch.cuda.memory_allocated('cuda:' + config['GPU']) / 1024 / 1024 / 1024,
                  "GB")
            if os.path.exists(root_to + 'train_dataset.pkl'):
                # warm_test, cold_test = get_warm_cold_split(samples)
                train_samples = load_pickle(root_to + 'train_dataset.pkl')
                val_samples = load_pickle(root_to + 'val_dataset.pkl')
                test_samples = load_pickle(root_to + 'test_dataset.pkl')
                train_dataset = train_samples
                val_dataset = val_samples
                test_dataset = test_samples
            else:
                train_dataset, val_dataset, test_dataset = split_data_bhc(
                    samples, config['RATIO'], (1 - config['RATIO']) / 2, (1 - config['RATIO']) / 2,
                    seed=config['SEED']
                )  # 这样似乎更快，固定随机种子的时候是一样的；
                save_pickle(train_dataset, root_to + 'train_dataset.pkl')
                save_pickle(val_dataset, root_to + 'val_dataset.pkl')
                save_pickle(test_dataset, root_to + 'test_dataset.pkl')
        finally:
            del reserved_tensor
            # torch.cuda.empty_cache() # 会突然爆oom

        endt = time.time()
        print('Train Dataset done!, Cost {} s'.format(endt - end))

    # STEP 2: load dataloader
    collate_fn = get_special_input(config)
    # missing_statistics(train_dataset, config)
    train_dataloader = get_dataloader(train_dataset, batch_size=config['BATCH'], shuffle=True, drop_last=True,
                                      collate_fn=collate_fn)
    # val_dataloader = get_dataloader(val_dataset, batch_size=config['BATCH']*5, shuffle=False, drop_last=True)
    test_dataloader = get_dataloader(test_dataset, batch_size=config['BATCH'], shuffle=False, drop_last=False,
                                     collate_fn=collate_fn)
    load_dataloader = time.time()
    print('Dataloader done!, Cost {} s'.format(load_dataloader - endt))

    # cache clear
    # torch.cuda.empty_cache() # 占用内存清理
    gc.collect()

    # STEP 3: create sft data
    map_dict = get_map_system(config)
    #create_sft_data_flask(train_dataloader, root_to + 'train_sft_data.jsonl', config, map_dict, task_mode=task_mode, train_mode=True, run_config=config['run_config'], topk=config['TOPK'])
    # create_sft_data_flask(val_dataloader, root_to + 'val_sft_data.pkl', config, map_dict, task_mode=task_mode, train_mode=True)
    #create_sft_data_flask(test_dataloader, root_to + 'test_sft_data.jsonl', config, map_dict, task_mode=task_mode, train_mode=False, run_config=config['run_config'], topk=config['TOPK']) # 注意，下面test中仅用pure，以便在inference进行
    # 
    # # Step 4: 按LLM Factory格式构建数据
    meta_path_dic = load_pickle('/hpc2hdd/home/xxxs349/xxxc/RAGHealth/data/ready/meta_path.pkl')
    meta_path_dic = {key: value for key, value in meta_path_dic.items() if key in map(str, range(0, 10))} # 和create_sft中保持一致。

    # 
    re_construct_format(root_to + 'train_sft_data.jsonl', root_to, meta_path_dic, train_mode=True)  # for sft
    # re_construct_format(root_to + 'train_sft_data.jsonl', root_to, meta_path_dic, train_mode=False)  # for sft
    re_construct_format(root_to + 'test_sft_data.jsonl',root_to,meta_path_dic, train_mode=False)

    # Step 5: 迁移数据到LLAMA Factory
    copy_file(src=root_to + 'all_data.jsonl',
              dst='all_data_{}_{}.jsonl'.format(config['DATASET'], config['TASK']))  # for sft; 如果我用all_data进行训练。
    copy_file(src=root_to + 'train_sft_data.jsonl', dst='train_sft_data_{}_{}.jsonl'.format(config['DATASET'], config[
        'TASK']))  # for ppo，不过是通过query去查找对应的path
    copy_file(src=root_to + 'test_sft_data.jsonl'.format(config['MODEL']), dst='test_sft_data_{}_{}.jsonl'.format(config['DATASET'], config[
        'TASK']), test_mode=True)  # 供查找golden_path/negative path, 感觉不要也罢
    copy_file(src=root_to + 'pure_path.jsonl',
              dst='pure_path_{}_{}.jsonl'.format(config['DATASET'], config['TASK']))  # for ppo，不过是通过query去查找对应的path




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Hi! RAGHealth!')
    task_fn, mode, task_mode = get_task_fn(config)

    train = False # 必须先train保证数据处理，模型训练完成
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
            evaluate_generation_flask(model_path, test_data_path, config, output_path,  task_mode=task_mode,run_config=config['run_config'], topk=config['TOPK'])
