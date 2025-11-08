#/bin/bash

source /dccstor/prinstructdata/miniconda3/etc/profile.d/conda.sh
conda activate codet

python add_unique_ids.py
#python starcoder2-self-align/src/star_align/ossinstruct_v4.py     --instruct_mode "C->I"     --seed_data_file text_chunks_v1.json     --tag concept_gen     --temperature 0.7     --model Qwen/Qwen2.5-Coder-7B-Instruct     --num_batched_requests 32     --seed_code_start_index 0     --num_sample_per_request 1     --model_type vllm

#python generate_completions_v1.py --model-type direct --model /dccstor/prinstructdata/llama-recipes/model_converted_HF/granite-3b-code-base-combined-canitedit-instructcoder --output-dir granite-3b-code-base-combined-canitedit-instructcoder-gen --completion-limit 20 --batch-size 1 --temperature 0.2 --top-p 0.95 --max-tokens 4096


