
import torch
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
)

base_model = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(
    base_model,
    cache_dir="./model_weights",
    use_auth_token='YOU_HUGGINGFACE_TOKEN',
)

model = LlamaForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir="./model_weights",
    use_auth_token='YOU_HUGGINGFACE_TOKEN',
)