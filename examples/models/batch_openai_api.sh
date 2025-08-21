MODEL="kimivl"
BENCHMARK="vsibench"
OUTPUT=result/${BENCHMARK}/${MODEL}_results.json

export OPENAI_API_BASE="http://127.0.0.1:8000/v1"
export OPENAI_API_KEY="any_string"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="~/.cache/huggingface"

python3 -m lmms_eval \
    --model batch_openai_compatible \
    --model_args model_version=${MODEL},azure_openai=False,max_frames_num=16,threads=256 \
    --tasks ${BENCHMARK}  \
    --batch_size 256 \
    --output_path ${OUTPUT} \
    --log_samples   \
    --log_samples_suffix ${MODEL}