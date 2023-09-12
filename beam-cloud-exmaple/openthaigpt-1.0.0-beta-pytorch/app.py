from beam import App, Runtime, Image, Output, Volume, VolumeType
import time
import torch
from transformers import (
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
)
from peft import PeftModel
import transformers
import json
import os.path as osp
from typing import Union

base_model = "ChanonUtupon/openthaigpt-merge-lora-llama-2-7B-3470k"
lora_weights = 'openthaigpt/openthaigpt-1.0.0-beta-7b-chat'


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        template_name = "alpaca"
        self.template = {
            "description": "Template used by Alpaca-LoRA.",
            "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
            "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
            "response_split": "### Response:"
        }
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


app = App(
    name="llama2",
    runtime=Runtime(
        cpu=4,
        memory="32Gi",
        gpu="A10G",
        image=Image(
            python_packages=[
                "accelerate",
                "transformers",
                "torch",
                "sentencepiece",
                "peft"
            ],
        ),
    ),
    volumes=[
        Volume(
            name="llama2-openthaigpt",
            path="./llama2-openthaigpt",
            volume_type=VolumeType.Persistent,
        )
    ],
)

def load_models():
    tokenizer = LlamaTokenizer.from_pretrained(
            base_model,
            cache_dir="./llama2-openthaigpt",
    )
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        cache_dir="./llama2-openthaigpt",
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        cache_dir="./llama2-openthaigpt",
        torch_dtype=torch.float16,
        device_map="auto",
    )

    return tokenizer, model


@app.rest_api(loader=load_models)
def generate(**inputs):
    prompt_template = ""
    prompter = Prompter(prompt_template)

    tokenizer, model = inputs["context"]
    # prompt = inputs["prompt"]
    text_input = inputs["prompt"]

    prompt = prompter.generate_prompt(text_input, None)

    load_8bit = False
    tokenizer.pad_token_id = 0  # unk
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2":
        model = torch.compile(model)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")

    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
    )
    start = time.time()
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=128,
            early_stopping=True,
            repetition_penalty=1,
            no_repeat_ngram_size=0
        )

    s = generation_output.sequences[0]
    decoded_output = tokenizer.decode(s).split("### Response:")[1].strip()
    return {"pred": decoded_output}