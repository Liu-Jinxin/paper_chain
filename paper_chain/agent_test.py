import yaml
import os
from transformers import AutoTokenizer
import transformers
import torch
from document_loaders.loader import load_documents


# Assuming you have the necessary URL, destination folder and chunk size
url = 'https://www.roboticsproceedings.org/rss19/p001.pdf'  # Replace this with your PDF URL
dest_folder = './data/pdfs'
chunk_size = 4000  # or any size you desire
overlap = 0  # optional parameter
max_chunks = 4  # optional parameter

chunks, filename = load_documents(url, dest_folder, chunk_size, overlap, max_chunks)
txt1 = "<s>[INST] <<SYS>> You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.<</SYS>>"
# Use the first chunk as input to your model
txt2 = "[/INST]"
first_chunk = txt1 + chunks[0] + " .Rephrase the content within 100 words. Don't cite references." + txt2 

# Your existing Hugging Face model code
with open('/workspace/paper_chain/config.yaml', 'r') as file:
    config = yaml.safe_load(file)
hf_token = config['HUGGINGFACE_TOKEN']

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)  # Removed use_auth_token here
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device='cuda:0'
)  # Removed use_auth_token here too

sequences = pipeline(
    first_chunk,  # Use the first chunk as input
    do_sample=True,
    top_k=50,
    temperature=1,
    #num_return_sequences=1,
    top_p=0.95,
    eos_token_id=tokenizer.eos_token_id,
    #max_length=1024,
    max_new_tokens=1024
)

for seq in sequences:
    all_part = seq['generated_text']
    new_part = seq['generated_text'].replace(first_chunk, "")  # Remove the original first_chunk from the generated text
    print("chunk", chunks[0])
    print("---------------------\n")
    print(f"All Generated Part: {all_part}")
    print("------------new part-----------")
    print(f"Newly Generated Part: {new_part}")
