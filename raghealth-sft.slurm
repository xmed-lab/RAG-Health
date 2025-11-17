#!/bin/bash
#SBATCH --job-name=sr-exp-rl-reason                # 作业名称
#SBATCH --array=0                           # 任务数组索引 (0, 共1个任务)
#SBATCH --nodes=1                             # 每个任务使用1个节点
#SBATCH --ntasks-per-node=1                   # 每个节点运行1个任务
#SBATCH --cpus-per-task=12                     # 每个任务分配12个CPU核心 , 超过12个核心有问题我发现
#SBATCH --gres=gpu:1                      # 每个节点使用4个GPU   
#SBATCH --mem=16G                             # 每个任务内存16GB ; ours可能需要更大，一般都是
#SBATCH --nodelist=gpu1-13,gpu1-18,gpu1-17,gpu1-18,gpu1-17,gpu1-14,gpu1-7,gpu1-8  # 指定任务节点; gpu1-8,gpu1-18,gpu1-17,gpu1-14,gpu1-7,gpu1-13
#SBATCH --partition=czhangcn_rent             # 指定分区
#SBATCH --output=experiment/exp_out/raghealth-ours-rea-miv-%A_%a.out  # 输出日志格式: TCGA-<主作业ID>_<子任务ID>.out
#SBATCH --error=experiment/exp_err/raghealth-ours-rea-miv-%A_%a.err   # 错误日志格式: TCGA-<主作业ID>_<子任务ID>.err

# CoT是qwq

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
echo "SLURM_LOCALID: " $SLURM_LOCALID
echo "SLURM_JOB_NODELIST:" $SLURM_JOB_NODELIST
echo "SLURM_PROCID: " $SLURM_PROCID
echo "SLURM_LOCALID:" $SLURM_LOCALID
echo "SLURM_NTASKS: " $SLURM_NTASKS
echo "SLURM_STEP_NODELIST: " $SLURM_STEP_NODELIST


# 激活虚拟环境
source /hpc2hdd/home/xxxs349/miniconda3/bin/activate ragllm

which python3

# 需要确定LLAMA dataset_info和examples有对应的东西

### 需要SFT的时候, 只需要1个GPU, 
# 预处理
#python3 /hpc2hdd/home/xxxs349/xxxc/RAGHealth/src/main.py #> /dev/null 2>&1 &
# 进入目录
cd /hpc2hdd/home/xxxs349/xxxc/RAGHealth/LLaMA-Factory
# 运行任务
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_lora/ours/REA/MIV/ours_lora_sft_custom.yaml # > sft_qwen25-7B_los.log 2>&1 &
# merge
llamafactory-cli export examples/merge_lora/ours/REA/MIV/ours_lora_sft_custom.yaml # 手动merge吧


