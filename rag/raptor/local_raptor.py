import os
import requests
from openai import OpenAI
from raptor import BaseSummarizationModel, BaseQAModel, BaseEmbeddingModel, RetrievalAugmentationConfig

class SummarizationModel(BaseSummarizationModel):
    def __init__(self):
        self.url = f'http://127.0.0.1:5000/v1/chat/completions'

    def summarize(self, context, max_tokens=150):
        # Format the prompt for summarization
        payload = {
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 50,
            "do_sample": True,
            "seed": 12345,
            "messages":[
                {"role": "user", "content": f"Write a summary of the following, including as many key details as possible: {context}:"}
            ] 
        }

        # Generate the summary using the pipeline
        response = requests.post(self.url, json=payload).json()    
        
        # Extracting and returning the generated summary
        summary = response['choices'][0]['message']['content'].strip()

        return summary

class QAModel(BaseQAModel):
    def __init__(self):
        # Initialize the tokenizer and the pipeline for the model
        self.url = f'http://127.0.0.1:5000/v1/chat/completions'

    def answer_question(self, context, question):
        # Apply the chat template for the context and question
        payload = {
            "max_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 50,
            "do_sample": True,
            "seed": 12345,
            "messages":[
              {"role": "user", "content": f"Given Context: {context} Give the best full answer amongst the option to question {question}"}
            ]
        }
        
        # Generate the answer using the pipeline
        response = requests.post(self.url, json=payload).json()  
        
        # Extracting and returning the generated answer
        # answer = outputs[0]["generated_text"][len(prompt):]
        answer = response['choices'][0]['message']['content'].strip()
        return answer

class EmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model="text-embedding-ada-002"):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_API_BASE"])
        self.model = model

    def create_embedding(self, text):
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(input=[text], model=self.model)
            .data[0]
            .embedding
        )

