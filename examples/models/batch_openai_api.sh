MODEL="kimi"
BENCHMARK="vsibench"
OUTPUT=result/${BENCHMARK}/${MODEL}_results.json

export OPENAI_API_KEY="xxx"
export OPENAI_API_BASE="http://xxx/v1"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="~/.cache/huggingface"

python3 -m lmms_eval \
    --model batch_openai_compatible \
    --model_args model_version=${MODEL},azure_openai=False,max_frames_num=16,threads=32 \
    --tasks ${BENCHMARK}  \
    --batch_size 32 \
    --output_path ${OUTPUT} \
    --log_samples   \
    --log_samples_suffix ${MODEL}