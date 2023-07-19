# combined_test.py
import os
from document_loaders.loader import load_documents
from document_question_answering.question_answering_roberta import Question_Answering

def main():
    # Define the PDF URL and destination folder
    pdf_url = 'https://www.roboticsproceedings.org/rss19/p001.pdf'  # Replace this with your PDF URL
    dest_folder = './data/pdfs'

    # Check if the destination folder exists, if not, create it
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    print("Destination folder checked/created.")
    
    # Load documents from the PDF URL and generate chunks
    chunks, filename = load_documents(pdf_url, dest_folder, chunk_size=1000, overlap=100)
    print(f"Document loaded and chunked. Filename: {filename}")
    
    # Create a Question_Answering object
    QA_model = Question_Answering()
    print("Question_Answering object created.")
    
    # Define the question
    question = 'What is the contribution?'
    print(f"Question defined: {question}")
    
    # Get the answers for the chunks
    answers = QA_model.get_answers_for_chunks(question, chunks)
    print("Answers retrieved for each chunk.")
    print(answers)
    
    # Filter and combine the answers
    final_answer = QA_model.filter_and_combine_top_answers(answers, n=5)
    print("Answers filtered and combined.")
    
    # Print the final answer
    print(f"Final answer: {final_answer}")

if __name__ == '__main__':
    main()
