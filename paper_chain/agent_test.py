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

# Use the first chunk as input to your model
first_chunk = chunks[0] + " Summarize the contribution of the above paper in 50 words or less."

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
    top_k=50,
    temperature=1,
    num_return_sequences=1,
    top_p=0.95,
    eos_token_id=tokenizer.eos_token_id,
    max_length=1024,
    max_new_tokens=1024
)

for seq in sequences:
    new_part = seq['generated_text'].replace(first_chunk, "")  # Remove the original first_chunk from the generated text
    print(f"Newly Generated Part: {new_part}")
