import os
from document_loaders.loader import load_documents
from document_question_answering.question_answering_llama2 import Question_Answering_Llama2

os.environ['PYDEVD_UNBLOCK_THREADS_TIMEOUT'] = '10'

def load_model_and_documents():
    # Define the PDF URL and destination folder
    pdf_url = 'https://www.roboticsproceedings.org/rss19/p001.pdf'  # Replace this with your PDF URL
    dest_folder = './data/pdfs'

    # Check if the destination folder exists, if not, create it
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    print("Destination folder checked/created.")
    
    # Load documents from the PDF URL and generate chunks
    chunks, filename = load_documents(pdf_url, dest_folder, chunk_size=4000, overlap=0, max_chunks=5)
    print(f"Document loaded and chunked. Filename: {filename}")
    
    # Create a Question_Answering_Llama2 object
    QA_model = Question_Answering_Llama2()
    print("Question_Answering_Llama2 object created.")
    
    return QA_model, chunks

def answer_question(QA_model, chunks, question, chunk_indices):
    # Print the question
    print(f"Question defined: {question}")

    # Show the selected chunks
    print(f"Selected chunks are:")
    for i in chunk_indices:
        print(f"Chunk {i}: {chunks[i]}")
    
    # Get the answers for the specified chunks
    answers = QA_model.get_answers_for_chunks(question, chunks, chunk_indices)
    
    # Combine the answers
    final_answer = QA_model.combine_answers(answers)
    print(f"Final answer: {final_answer}")

if __name__ == '__main__':
    QA_model, chunks = load_model_and_documents()
    question = 'What is the contribution for the above paper?'  # Define your question here
    chunk_indices = [0]  # Specify the chunks you want to process here
    print("Model and documents loaded. Ready to answer the question.")
    print("Chunks are:", chunks)
    answer_question(QA_model, chunks, question, chunk_indices)

