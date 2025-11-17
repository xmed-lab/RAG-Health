#!/bin/bash
#SBATCH --job-name=sr-exp-rl-reason                # 作业名称
#SBATCH --array=0                           # 任务数组索引 (0, 共1个任务)
#SBATCH --nodes=1                             # 每个任务使用1个节点
#SBATCH --ntasks-per-node=1                   # 每个节点运行1个任务
#SBATCH --cpus-per-task=12                     # 每个任务分配12个CPU核心 , 超过12个核心有问题我发现
#SBATCH --gres=gpu:2                     # 每个节点使用4个GPU   
#SBATCH --mem=16G                             # 每个任务内存16GB ; ours可能需要更大，一般都是
#SBATCH --nodelist=gpu1-8,gpu1-18,gpu1-14,gpu1-7,gpu1-13,gpu1-17  # 指定任务节点; !!!!!!!注意同一个节点端口会被占用。
#SBATCH --partition=czhangcn_rent             # 指定分区
#SBATCH --output=experiment/exp_out/raghealth-ours-rea-miv-test-%A_%a.out  # 输出日志格式: TCGA-<主作业ID>_<子任务ID>.out main尾缀是拒绝采样；test是最后的test.
#SBATCH --error=experiment/exp_err/raghealth-ours-rea-miv-test-%A_%a.err   # 错误日志格式: TCGA-<主作业ID>_<子任务ID>.err

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
#llamafactory-cli train examples/train_lora/ours/LOS/eICU/qwen25-7B_lora_sft_custom.yaml # > sft_qwen25-7B_los.log 2>&1 &
# merge
# llamafactory-cli export examples/merge_lora/ours/LOS/MIII/ours_lora_sft_custom.yaml


# 上面搞完之后，修改utils->port； config: path; main; GPU改为4, 建议还是改作业名称，不然根本不知道是哪一个 (下面的有main)

# nohup python3 -m vllm.entrypoints.openai.api_server --model /hpc2hdd/home/xxxs349/xxxc/RAGHealth/LLaMA-Factory/output/MOR/MIII/qwen25-7B_lora_sft --host 0.0.0.0 --port 8050 --dtype half --max-num-seqs 32 --gpu-memory-utilization 0.8 --max-model-len 20000 --tensor-parallel-size 2 --trust-remote-code > llm2_log.log 2>&1 &
#nohup python3 /hpc2hdd/home/xxxs349/xxxc/RAGHealth/src/rag_flask.py > rag.log 2>&1 &

# nohup gunicorn -w 2 -b 0.0.0.0:5000 rag_flask:app > rag_debug.log 2>&1 & #


###需要分布式测评的时候, 最好保证在一个集群上， 注意同一个集群端口会互相占用
export CUDA_VISIBLE_DEVICES=0,1,2,3

#nohup python3 -m vllm.entrypoints.openai.api_server --model /hpc2hdd/home/xxxs349/xxxc/huggingface/hub/qwen25-7B --host 0.0.0.0 --port 8890 --dtype half --max-num-seqs 32 --gpu-memory-utilization 0.6 --max-model-len 30000 --tensor-parallel-size 4 --trust-remote-code > llm_debugxx.log 2>&1 &

nohup python3 -m vllm.entrypoints.openai.api_server --model /hpc2hdd/home/xxxs349/xxxc/RAGHealth/LLaMA-Factory/output/LOS/MIV/ours_lora_sft --host 0.0.0.0 --port 8870 --dtype half --max-num-seqs 32 --gpu-memory-utilization 0.6 --max-model-len 30000 --tensor-parallel-size 2 --trust-remote-code > llm_debug_new.log 2>&1 &

sleep 60


# 如果是同一个节点，记得换端口, 而且不要共用一个输出
# nohup python3 -m vllm.entrypoints.openai.api_server --model /hpc2hdd/home/xxxs349/xxxc/RAGHealth/LLaMA-Factory/output/MOR/MIV/lightrag_lora_reward --host 0.0.0.0 --port 8000 --dtype half --max-num-seqs 32 --gpu-memory-#utilization 0.8 --max-model-len 30000 --tensor-parallel-size 4 --trust-remote-code > llm_debug.log 2>&1 &


# 等待服务启动（可通过端口检查或日志关键字判断）
echo "等待LLM服务启动..."
while true; do
    # 方法1：检查端口是否开放
    if nc -z localhost 8870; then
        echo "8870"
        break
    fi
    
    # 避免CPU占用过高
    sleep 1
done

cd /hpc2hdd/home/xxxs349/xxxc/RAGHealth/src
nohup gunicorn -w 2 -k gevent -b 0.0.0.0:5000 rag_flask:app > rag_debugxx.log 2>&1 & #


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


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

export RAY_memory_monitor_refresh_ms=0 # 我又开了，xs。


# 查看到底tmd是flask超过负载还是ray超过负载
# python3 /hpc2hdd/home/xxxs349/xxxc/RAGHealth/src/main.py # > medrag_mor_miv.log 2>&1 &
python3 /hpc2hdd/home/xxxs349/xxxc/RAGHealth/src/main.py # > medrag_mor_miv.log 2>&1 &

