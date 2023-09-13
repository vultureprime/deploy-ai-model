from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import time
print(transformers.__version__,torch.__version__)

model_id = "tiiuae/falcon-40b-instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    cache_dir="./falcon-40b",

)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    load_in_8bit=True,
    device_map="auto",
    cache_dir="./falcon-40b",
)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

start = time.time()
sequences = pipeline(
   "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
print(time.time() - start )
for seq in sequences:
    print(f"Result: {seq['generated_text']}")