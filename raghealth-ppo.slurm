#!/bin/bash
#SBATCH --job-name=sr-exp-rl-reason                # 作业名称
#SBATCH --array=0                           # 任务数组索引 (0, 共1个任务)
#SBATCH --nodes=1                             # 每个任务使用1个节点
#SBATCH --ntasks-per-node=1                   # 每个节点运行1个任务
#SBATCH --cpus-per-task=12                     # 每个任务分配12个CPU核心 , 超过12个核心有问题我发现
#SBATCH --gres=gpu:2                       # 每个节点使用4个GPU   
#SBATCH --mem=16G                             # 每个任务内存16GB ; ours可能需要更大，一般都是
#SBATCH --nodelist=gpu1-8,gpu1-18,gpu1-14,gpu1-7,gpu1-13  # 指定任务节点; !!!!!!!注意同一个节点端口会被占用。
#SBATCH --partition=czhangcn_rent             # 指定分区
#SBATCH --output=experiment/exp_out/raghealth-ours-rea-miv-ppo-%A_%a.out  # 输出日志格式: TCGA-<主作业ID>_<子任务ID>.out main尾缀是拒绝采样；test是最后的test.
#SBATCH --error=experiment/exp_err/raghealth-ours-rea-miv-ppo-%A_%a.err   # 错误日志格式: TCGA-<主作业ID>_<子任务ID>.err; 改一次任务LLAMA得重装一次？

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

cd /hpc2hdd/home/xxxs349/xxxc/RAGHealth/src
nohup gunicorn -w 4 -k gevent -b 0.0.0.0:5000 rag_flask:app > rag_debug.log 2>&1 & #


# 这个仅适用于think之后的benchmark，
echo "等待RAG服务启动..."
while true; do
    # 先检查端口是否开放
    if nc -z localhost 5000; then
        # 再尝试访问健康检查端点
        if curl -s http://localhost:5000/health | grep -q "OK"; then
            echo "服务已完全启动并就绪"
            break
        else
            echo "服务端口已开放，但应用尚未完全就绪，继续等待..."
        fi
    fi
    
    sleep 1
done

# 现在可以安全地运行benchmark
echo "开始执行benchmark测试..."

sleep 60


export CUDA_VISIBLE_DEVICES=0,1,2,3

export RAY_memory_monitor_refresh_ms=0 # 我又开了，xs。


# 注意修改trainer_multi, dataset_info
cd /hpc2hdd/home/xxxs349/xxxc/RAGHealth/LLaMA-Factory

# pip install --no-deps -e .

llamafactory-cli train examples/train_lora/ours/REA/MIV/ours_lora_ppo_custom.yaml # 要和DEFAULT CONFIG一致
llamafactory-cli export examples/merge_lora/ours/REA/MIV/ours_lora_ppo_custom.yaml

