import yaml
from transformers import AutoTokenizer, pipeline
import torch

class Question_Answering_Llama2():
    def __init__(self) -> None:
        self.model_name = "meta-llama/Llama-2-7b-chat-hf"
        with open('/home/jinxin/Desktop/github/paper_chain/paper_chain/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.hf_token = config['HUGGINGFACE_TOKEN']
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_auth_token=self.hf_token)
        self.qa_model = pipeline('text-generation', model=self.model_name, tokenizer=self.tokenizer,
                                 torch_dtype=torch.float16, device='cuda:0', use_auth_token=self.hf_token)

    def get_answer(self, question):
        # Use the model to get the answer for a single question
        res = self.qa_model(question, temperature=1, num_return_sequences=1, top_p=0.95, 
                            eos_token_id=self.tokenizer.eos_token_id, max_length=1024, max_new_tokens=1024)
        return res[0]['generated_text']

    def get_answers_for_chunks(self, question, chunks, chunk_indices):
        # Get the answer for a single question and specified contexts (chunks)
        answers = []
        for i in chunk_indices:
            chunk = chunks[i]
            answer = self.get_answer(chunk + ' ' + question)
            answers.append(answer)
        return answers

    def combine_answers(self, answers):
        # Combine the answers into a single string
        final_answer = ' '.join(answers)
        return final_answer

if __name__ == '__main__':
    # Create a Question_Answering_Llama2 object
    QA_model = Question_Answering_Llama2()
    # Define the question and the chunks
    chunks = [
        'I liked "Breaking Bad".',
        'I liked "Band of Brothers".',
    ]
    question = 'Do you have any recommendations of other shows I might like?'
    # Get the answers for the chunks
    answers = QA_model.get_answers_for_chunks(question, chunks)
    # Combine the answers
    final_answer = QA_model.combine_answers(answers)
    # Print the final answer
    print(final_answer)
