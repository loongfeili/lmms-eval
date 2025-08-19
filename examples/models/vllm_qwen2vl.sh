
# pip3 install vllm
# pip3 install qwen_vl_utils

# cd ~/prod/lmms-eval-public
# pip3 install -e .
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=18000000
export NCCL_DEBUG=DEBUG
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export HF_HOME="./.cache/huggingface"

python3 -m lmms_eval \
    --model vllm \
    --model_args model_version=/opt/data/private/llf/3dmllm/moonshotai/Kimi-VL-A3B-Instruct,tensor_parallel_size=8,gpu_memory_utilization=0.95,max_frame_num=16,max_model_len=16384,limit_mm_per_prompt='{"image":16}',max_num_seqs=2\
    --tasks vsibench \
    --batch_size 2 \
    --log_samples \
    --log_samples_suffix vllm \
    --output_path ./logs 