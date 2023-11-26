
import os
import torch
from transformers import (
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
)

base_model = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(
    base_model,
    cache_dir="./tokenizer",
    use_auth_token='YOU_HUGGINGFACE_TOKEN',
)