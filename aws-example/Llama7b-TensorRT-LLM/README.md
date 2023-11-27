# Llama7b-TensorRT-LLM

#### Tutorial by [![N|Solid](https://vultureprime-research-center.s3.ap-southeast-1.amazonaws.com/vulturePrimeLogo.png)](https://vultureprime.com)

## For English Version Please visit
[English Version](https://github.com/vultureprime/deploy-ai-model/blob/main/aws-example/Llama7b-TensorRT-LLM/README_eng.md)

## สิ่งที่ต้องมี
- [AWS](https://aws.amazon.com/) Account 

## GPU Requirement 
- VRAM 24GB (A10g)
- Stroage 500 GB

## ขั้นตอน 
1. เข้าใช้งาน EC2 
2. สร้าง Virtual Machine โดยเลือกใช้งาน EC2 รุ่น G5.xlarge เลือก Image เป็น Ubuntu AMI Deep learning Base GPU AMI (ubuntu 20.04) 20231026 AMI ID : ami-03b036db02e8a6443
> การสมัครใช้งานครั้งแรก จะไม่สามารถเลือก GPU ได้ จำเป็นต้องนัดหมายคุยกับทีม Support ของทาง AWS เพื่อขอเปิดใช้งาน GPU

3. ติดตั้ง เพิ่มพื้นที่ Swap สำหรับ Build Model  
```
fallocate -l 128G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile && free -h
```
4. Download TensorRT-LLM

```
git clone https://github.com/NVIDIA/TensorRT-LLM.git && cd TensorRT-LLM 
```

~4.1 แก้ไข ชั่วคราว ทางทีมพัฒนา TensorRT-LLM ได้แก้ไขไฟล์ Docker แล้ว แต่ยังไม่ได้ทำการ Release ดังนั้นจึงต้อง
ทำการแก้ไขไฟล์ Docker ใน TensorRT-LLM/docker/common/install_bash.sh [Issue447](https://github.com/NVIDIA/TensorRT-LLM/issues/447)
ให้เป็นดังนี้~

แก้ไขเรียบร้อย [a21e2f8](https://github.com/NVIDIA/TensorRT-LLM/commit/a21e2f85178111fed9812bb88c2cc7411b25f0ba)

```
#pip install mpi4py
pip install git+https://github.com/Shixiaowei02/mpi4py.git@fix-setuptools-version
```

5. ติดตั้ง TensorRT-LLM 
หมายเหตุ : การติดตั้ง TensorRT-LLM จะใช้เวลานาน โดยเฉลี่ยใช้เวลา 30-60 นาที และจะมีการติดตั้ง Library ต่างๆ อัตโนมัติ 
Parameter  ``` CUDA_ARCHS="86-real" ```  บ่งบอกถึง GPU-Architecture ที่ต้องการติดตั้ง ซึ่ง A10G อยู่ใน SM 86 ดังนั้นจึงใช้คำสั่งดังนี้ 
ถ้าหากต้องการติดตั้งบน GPU อื่นๆ สามารถดู GPU-Architecture (Compute Capability) ได้จากลิงค์นี้ [GPU-Architecture](https://developer.nvidia.com/cuda-gpus) และเปลี่ยน Parameter ดังกล่าวตามต้องการ

ถ้าไม่ใส่ Parameter ``` CUDA_ARCHS ``` ดังกล่าวจะเป็นการ Build ทุก Architecture ที่รองรับ ซึ่งจะใช้เวลานานกว่า 1 ชั่วโมง

```
apt-get update && apt-get -y install git git-lfs && git submodule update --init --recursive && git lfs install && git lfs pull && make -C docker release_build CUDA_ARCHS="86-real" && make -C docker release_run
```

6. เมื่อเข้ามาใน docker และวให้เข้าไปยัง Folder examples/llama
```
cd /code/tensorrt_llm/
```
7. ให้ทำการ Load Llama Model ด้วยโค้ดชุดนี้ load_model.py
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

8. ทำการ Build Llama Model ด้วยโค้ดชุดนี้ build.py
Parameter ที่สำคัญสามารถดูได้จาก [TensorRT-LLM Llama](https://github.com/NVIDIA/TensorRT-LLM/tree/release/0.5.0/examples/llama)
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

9. ทำการ ทดสอบ Llama Model ด้วยโค้ดชุดนี้ test.py
สามารถเปลี่ยน Input Text ได้จาก Parameter ```--input_text```
```
python3 run.py --engine_dir=/code/tensorrt_llm/examples/llama/tmp_64/llama/7B/trt_engines/weight_only/1-gpu/ --max_output_len 100 --tokenizer_dir model_weights/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93 --input_text "Hello how are you?"
```

10. ทดสอบ Benchmark ได้ด้วย โค้ดชุดนี้ benchmark.py

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

11. Build สำหรับ Prod ด้วยโค้ดชุดนี้ build.py
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

12. เปิด Terminal Session ใหม่ และทำการ Run ด้วยโค้ดชุดนี้
```
git clone -b release/0.5.0 https://github.com/triton-inference-server/tensorrtllm_backend.git && cd tensorrtllm_backend
```

13. ทำการ Copy ไฟล์ Engine ที่ Build ไว้ในข้อ 11 มาไว้ที่ Folder tensorrtllm_backend
```
docker cp {YOUR_DOCKER_ID}:/code/tensorrt_llm/examples/llama/tmp_prod/llama/7B/trt_engines/weight_only/1-gpu/ all_models/inflight_batcher_llm/tensorrt_llm/1/
```

14. ทำการ Config ไฟล์ต่าง ๆ ด้วยโค้ดชุดนี้

```
python3 tools/fill_template.py --in_place \
      all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt \
      decoupled_mode:true,engine_dir:/all_models/inflight_batcher_llm/tensorrt_llm/1/1-gpu,\
max_tokens_in_paged_kv_cache:5120,batch_scheduler_policy:guaranteed_completion,kv_cache_free_gpu_mem_fraction:0.4,\
max_num_sequences:32
```

Tokenizer สำหรับ Encode
```
python3 tools/fill_template.py --in_place \
    all_models/inflight_batcher_llm/preprocessing/config.pbtxt \
    tokenizer_type:llama,tokenizer_dir:/opt/tritonserver/tokenizer/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93/
```

Tokenizer สำหรับ Decode
```
 
python3 tools/fill_template.py --in_place \
    all_models/inflight_batcher_llm/postprocessing/config.pbtxt \
    tokenizer_type:llama,tokenizer_dir:/opt/tritonserver/tokenizer/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93/
```

15. ทำการ Run ด้วยโค้ดชุดนี้
```
sudo docker run -it --rm --gpus all --network host --shm-size=1g \
-v $(pwd)/all_models:/all_models \
-v $(pwd)/scripts:/opt/scripts \
nvcr.io/nvidia/tritonserver:23.10-trtllm-python-py3
```

16. Install Library ที่จำเป็น
```
pip install sentencepiece protobuf
```

17. โหลด Tokenizer ด้วยโค้ดชุดนี้
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

18. จากนั้นทำการ Run Triton server

```
python /opt/scripts/launch_triton_server.py --model_repo /all_models/inflight_batcher_llm --world_size 1
```

19. ทดสอบการทำงานด้วยโค้ดชุดนี้
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
