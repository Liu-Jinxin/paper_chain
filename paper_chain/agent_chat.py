from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import yaml

print('cuda_version:', torch.version.cuda)

with open('/workspace/paper_chain/config.yaml', 'r') as file:
    config = yaml.safe_load(file)
hf_token = config['HUGGINGFACE_TOKEN']

model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', device_map='auto', torch_dtype=torch.float16, use_auth_token=hf_token, load_in_8bit=True)
model =model.eval()
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf', use_fast=False, use_auth_token=hf_token)
tokenizer.pad_token = tokenizer.eos_token
input_ids = tokenizer(['<s>Human: how to control robot arm?\n</s><s>Assistant: '], return_tensors="pt",add_special_tokens=False).input_ids.to('cuda')        
generate_input = {
    "input_ids":input_ids,
    "max_new_tokens":512,
    "do_sample":True,
    "top_k":50,
    "top_p":0.95,
    "temperature":1,
    "repetition_penalty":1.3,
    "eos_token_id":tokenizer.eos_token_id,
    "bos_token_id":tokenizer.bos_token_id,
    "pad_token_id":tokenizer.pad_token_id
}
generate_ids  = model.generate(**generate_input)
text = tokenizer.decode(generate_ids[0])
print(text)