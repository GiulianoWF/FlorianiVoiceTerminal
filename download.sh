#!/bin/bash
source venv_gguf/bin/activate

pip install huggingface_hub

python -m huggingface_hub login

# Download Q4_K_M (~19.8GB, more room for KV cache)
python -c "

from huggingface_hub import hf_hub_download
hf_hub_download('bartowski/Qwen_Qwen3-32B-GGUF',
    filename='Qwen_Qwen3-32B-Q4_K_M.gguf',
    local_dir='./models/qwen3-32b')

"
