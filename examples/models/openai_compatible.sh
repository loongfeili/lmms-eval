# export http_proxy=http://127.0.0.1:7890
# export https_proxy=http://127.0.0.1:7890
# MODEL="doubao-1.5-vision-pro-250328"
MODEL="Qwen/Qwen2.5-VL-72B-Instruct"
BENCHMARK=""
OUTPUT=result/${BENCHMARK}/${MODEL}_results.json
# export OPENAI_API_KEY="2a1bb43f-1716-4337-9e14-5a3e92eb95ea"
# export OPENAI_API_BASE="https://ark.cn-beijing.volces.com/api/v3"
export OPENAI_API_KEY="sk-drcaiymsoxcpibagipqvrfcitgmbvgvzqcaqehqdudwntqtl"
export OPENAI_API_BASE="https://api.siliconflow.cn/v1"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="~/.cache/huggingface"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
python3 -m lmms_eval \
    --model openai_compatible \
    --model_args model_version=${MODEL},azure_openai=False \
    --tasks ${BENCHMARK}  \
    --batch_size 1 \
    --output_path ${OUTPUT} \
    --log_samples   \
    --log_samples_suffix _${MODEL}