import yaml
import os
from transformers import AutoTokenizer
import transformers
import torch

print(torch.version.cuda)

with open('/home/jinxin/Desktop/github/paper_chain/paper_chain/config.yaml', 'r') as file:
    config = yaml.safe_load(file)
hf_token = config['HUGGINGFACE_TOKEN']

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=hf_token)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device='cuda:0',
    use_auth_token=hf_token
)

sequences = pipeline(
    'chunks',
    top_k=50,
    temperature=1,
    num_return_sequences=1,
    top_p=0.95,
    eos_token_id=tokenizer.eos_token_id,
    max_length=1024,
    max_new_tokens=1024
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
