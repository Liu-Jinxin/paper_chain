import requests
from sentence_transformers.util import semantic_search
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import personal


class Retrieval_From_Database():
    def __init__(self) -> None:
        self.database = None
        self.database_embeddings_list = None
        self.model_id = "sentence-transformers/all-MiniLM-L6-v2"
        self.hf_token = personal.hf_token
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_id}"
        self.headers = {"Authorization": f"Bearer {self.hf_token}"}

    
    def update_database(self, raw_texts, list_of_chunks=None, add=False):
        """
        Update the database
        
        :param raw_texts: a list of texts.
        :param list_of_chunks: each element in the list is a list of text chunks
        """        
        output = []
        if list_of_chunks is None:
            embeddings = self.embedding(raw_texts)
            for e in embeddings:
                output.append([e]) # each embedding is regarded as a list 
        else:
            for chunk in list_of_chunks:
                embeddings = self.embedding(raw_texts)
                output.append(embeddings)            
        #print(output)
        if add:
            if self.database is None:
                self.database_embeddings_list = output
                self.database = raw_texts
            else:
                self.database_embeddings_list.extend(output)
                self.database.extend(raw_texts)
        else:
            self.database_embeddings_list = output
            self.database = raw_texts
    
    def embedding(self, text):
        response = requests.post(self.api_url, headers=self.headers, json={"inputs": texts, "options":{"wait_for_model":True}})
        return response.json()

    def retrieval(self, query, K=1):
        output = self.embedding(query)
        query_embedding = torch.FloatTensor(output)
        scores = []
        # for each document, taking the highest score among all the document chunks
        for database_embeddings in self.database_embeddings_list:
            cos_scores = util.cos_sim(query_embedding, database_embeddings)[0]
            top_results = torch.topk(cos_scores, k=1)        
            scores.append(top_results[0])
        
        top_results = torch.topk(torch.tensor(scores), k=K)
        text_results = [self.database[i] for i in top_results[1]]
        return top_results[0], text_results
        '''
        hits = semantic_search(query_embeddings, torch.tensor(np.array(self.database_embeddings)).float(), top_k=K)
        results = []
        for i in range(len(hits[0])):
            results.append(self.database[hits[0][i]['corpus_id']])
        return hits, results
        '''




if __name__ == '__main__':
    texts = ["How do I get a replacement Medicare card?",
        "What is the monthly premium for Medicare Part B?",
        "How do I terminate my Medicare Part B (medical insurance)?",
        "How do I sign up for Medicare?",
        "Can I sign up for Medicare Part B if I am working and have health insurance through an employer?",
        "How do I sign up for Medicare Part B if I already have Part A?",
        "What are Medicare late enrollment penalties?",
        "What is Medicare and who can get it?",
        "How can I get help with my Medicare Part A and Part B premiums?",
        "What are the different parts of Medicare?",
        "Will my Medicare premiums be higher because of my higher income?",
        "What is TRICARE ?",
        "Should I sign up for Medicare Part B if I have Veterans' Benefits?"]
    query = "How to get a alternative Medicare card?"

    retrieval = Retrieval_From_Database()
    retrieval.update_database(texts)
    scores, results = retrieval.retrieval(query, K=3)
    print(scores)
    print(results)
    
