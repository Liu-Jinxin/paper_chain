from transformers import pipeline


class Question_Answering():
    def __init__(self) -> None:
        self.model_name = "deepset/roberta-base-squad2"
        self.qa_model = pipeline('question-answering', model=self.model_name, tokenizer=self.model_name)

    def get_answer(self, QA_input):
        res = self.qa_model(QA_input)
        return res

if __name__ == '__main__':
    QA_model = Question_Answering()
    QA_input = {
        'question': 'Why is model conversion important?',
        'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
    }
    print(QA_model.get_answer(QA_input))