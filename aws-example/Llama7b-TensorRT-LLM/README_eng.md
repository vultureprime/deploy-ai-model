# Llama7b-TensorRT-LLM

#### Tutorial by [![N|Solid](https://vultureprime-research-center.s3.ap-southeast-1.amazonaws.com/vulturePrimeLogo.png)](https://vultureprime.com)

## Pre-requisite
- [AWS](https://aws.amazon.com/) Account 

## GPU Requirement 
- VRAM 24GB (A10g)
- Stroage 500 GB

## Steps 
1. Login to AWS Console and go to EC2.
2. Create Virtual Machine with G5.xlarge Instance Type and select Image as Ubuntu AMI Deep learning Base GPU AMI (ubuntu 20.04) 20231026 AMI ID : ami-03b036db02e8a6443
> First time user will not be able to select GPU Instance Type, you need to contact AWS Support to enable GPU Instance Type

3. Install Swap Space for Build Model  
```
fallocate -l 128G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile && free -h
```

4. Download TensorRT-LLM.
```
git clone https://github.com/NVIDIA/TensorRT-LLM.git && cd TensorRT-LLM 
```

4.1 Temporary Fix TensorRT-LLM Team has fixed the Dockerfile but not yet release, so we need to fix it manually by edit TensorRT-LLM/docker/common/install_bash.sh
[Issue447](https://github.com/NVIDIA/TensorRT-LLM/issues/447)

```
#pip install mpi4py
pip install git+https://github.com/Shixiaowei02/mpi4py.git@fix-setuptools-version
```


5. Install TensorRT-LLM.
Remark : TensorRT-LLM Installation will take time around 30-60 minutes and it will install all required library automatically.
Parameter  ``` CUDA_ARCHS="86-real" ```  is GPU-Architecture that you want to install, A10G is SM 86 so we use this parameter.
If you want to install on other GPU, you can check GPU-Architecture (Compute Capability) from this link [GPU-Architecture](https://developer.nvidia.com/cuda-gpus) and change Parameter as you want.

If you don't use Parameter ``` CUDA_ARCHS ``` TensorRT-LLM will install all GPU-Architecture that support TensorRT-LLM and it will take more than 1 hour.

```
apt-get update && apt-get -y install git git-lfs && git clone https://github.com/NVIDIA/TensorRT-LLM.git && cd TensorRT-LLM && git submodule update --init --recursive && git lfs install && git lfs pull && make -C docker release_build CUDA_ARCHS="86-real" && make -C docker release_run
```

6. After enter docker and go to examples/llama.
```
cd /code/tensorrt_llm/
```
7. Load Llama2-7b Model with this code load_model.py.
```
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
```
8. Build Llama Model with this code build.py.
You could visit parameter configuration from [TensorRT-LLM Llama](https://github.com/NVIDIA/TensorRT-LLM/tree/release/0.5.0/examples/llama)
FP16 
```
python build.py --model_dir model_weights/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93 \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --max_batch_size 16 \
                --enable_context_fmha \
                --output_dir ./tmp_64/llama/7B/trt_engines/weight_only/1-gpu/
```
Int4

```
python build.py --model_dir model_weights/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93 \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --use_weight_only \
                --weight_only_precision 'int4'\
                --output_dir ./tmp_64/llama/7B/trt_engines/weight_only/1-gpu/
```
9. Test Llama Model with this code test.py.
You could change Input Text from Parameter ```--input_text```
```
python3 run.py --engine_dir=/code/tensorrt_llm/examples/llama/tmp_64/llama/7B/trt_engines/weight_only/1-gpu/ --max_output_len 100 --tokenizer_dir model_weights/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93 --input_text "Hello how are you?"
```

10. Benchmark with this code benchmark.py.

```
cd /code/tensorrt_llm/benchmarks/python
```

```
python benchmark.py \
    --mode plugin \
    -m llama_7b \
    --engine_dir "/code/tensorrt_llm/examples/llama/tmp_64/llama/7B/trt_engines/weight_only/1-gpu/" \
    --batch_size "1;10;20;30;60;120;240" \
    --csv \
    --input_output_len "32,128;32,256;32,512;32,1024;128,128;128,256;128,512;128,1024;256,128;256,256;256,512;256,1024;512,128;512,256;512,512;512,1024;1024,128;1024,256;1024,512;1024,1024"
```

11. Build for Prod with this code.
```
cd /code/tensorrt_llm/
```

```
python build.py --model_dir model_weights/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93 \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --max_batch_size 32 \
                --use_inflight_batching \
                --paged_kv_cache \
                --output_dir ./tmp_prod/llama/7B/trt_engines/weight_only/1-gpu/
```
12. Open New Terminal Session and run this code to install Triton Server.

```
git clone -b release/0.5.0 https://github.com/triton-inference-server/tensorrtllm_backend.git && cd tensorrtllm_backend
```

13. Copy Engine file from Step 11 to tensorrtllm_backend Folder.

```
docker cp {YOUR_DOCKER_ID}:/code/tensorrt_llm/examples/llama/tmp_prod/llama/7B/trt_engines/weight_only/1-gpu/ all_models/inflight_batcher_llm/tensorrt_llm/1/
```

14. Config File with this code.
```
python3 tools/fill_template.py --in_place \
      all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt \
      decoupled_mode:true,engine_dir:/all_models/inflight_batcher_llm/tensorrt_llm/1/1-gpu,\
max_tokens_in_paged_kv_cache:5120,batch_scheduler_policy:guaranteed_completion,kv_cache_free_gpu_mem_fraction:0.4,\
max_num_sequences:32
```

Tokenizer for Encode
```
python3 tools/fill_template.py --in_place \
    all_models/inflight_batcher_llm/preprocessing/config.pbtxt \
    tokenizer_type:llama,tokenizer_dir:/opt/tritonserver/tokenizer/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93/
```

Tokenizer for Decode
```
python3 tools/fill_template.py --in_place \
    all_models/inflight_batcher_llm/postprocessing/config.pbtxt \
    tokenizer_type:llama,tokenizer_dir:/opt/tritonserver/tokenizer/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93/
```

15. Run with this code.
```
sudo docker run -it --rm --gpus all --network host --shm-size=1g \
-v $(pwd)/all_models:/all_models \
-v $(pwd)/scripts:/opt/scripts \
nvcr.io/nvidia/tritonserver:23.10-trtllm-python-py3
```

16. Install Library that required.
```
pip install sentencepiece protobuf
```

17. Download Tokenizer with this code.
```
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
```

18. Run Triton server

```
python /opt/scripts/launch_triton_server.py --model_repo /all_models/inflight_batcher_llm --world_size 1
```

19. Test with this code.
```
curl --noproxy '*' -X POST localhost:8000/v2/models/ensemble/generate -d \
'{
"text_input": "Hello how are you?",
"parameters": {
"max_tokens": 100,
"bad_words":[""],
"stop_words":[""],
"stream": false,
"temperature": 1
}
}'
```

## License 
MIT
