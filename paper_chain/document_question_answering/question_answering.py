from transformers import pipeline

class Question_Answering():
    def __init__(self) -> None:
        # Initialize the model and tokenizer
        self.model_name = "deepset/roberta-base-squad2"
        self.qa_model = pipeline('question-answering', model=self.model_name, tokenizer=self.model_name)

    def get_answer(self, QA_input):
        # Use the model to get the answer for a single question and context
        res = self.qa_model(QA_input)
        return res

    def get_answers_for_chunks(self, question, chunks):
        # Get the answer for a single question and multiple contexts (chunks)
        answers = []
        for chunk in chunks:
            QA_input = {
                'question': question,
                'context': chunk
            }
            answer = self.get_answer(QA_input)
            answers.append(answer)
        return answers

    def filter_and_combine_answers(self, answers, score_threshold=0.2):
        # Filter the answers based on the score threshold and combine them into a single string
        filtered_answers = [answer['answer'] for answer in answers if answer['score'] > score_threshold]
        final_answer = ' '.join(filtered_answers)
        return final_answer

if __name__ == '__main__':
    # Create a Question_Answering object
    QA_model = Question_Answering()
    # Define the question and the chunks
    question = 'Why is model conversion important?'
    chunks = [
        'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.',
        'Another chunk of text.',
        'And another chunk of text.'
    ]
    # Get the answers for the chunks
    answers = QA_model.get_answers_for_chunks(question, chunks)
    # Filter and combine the answers
    final_answer = QA_model.filter_and_combine_answers(answers)
    # Print the final answer
    print(final_answer)
