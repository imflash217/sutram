import pandas as pd
import numpy as np
import torch
from transformers import pipeline

## ---------------------------------------------------------------------------------##

TEXT = """Dear Amazon, last week I ordered an Optimus Prime action figure
from your online store in Germany. Unfortunately, when I opened the package,
I discovered to my horror that I had been sent an action figure of Megatron
instead! As a lifelong enemy of the Decepticons, I hope you can understand my
dilemma. To resolve the issue, I demand an exchange of Megatron for the
Optimus Prime figure I ordered. Enclosed are copies of my records concerning
this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""

## ---------------------------------------------------------------------------------##


def text_classification(text):
    classifier = pipeline("text-classification")
    outputs = classifier(text)
    
    print("---" * 30, "\n", "Text Classification", "\n", "---"*10)
    print(pd.DataFrame(outputs))
    print("---" * 30)


def named_entity_recognition(text):
    ner_tagger = pipeline("ner", aggregation_strategy="simple")
    outputs = ner_tagger(text)

    print("---" * 30, "\n", "Named Entity Recognition", "\n", "---"*10)
    print(pd.DataFrame(outputs))
    print("---" * 30)

def question_answering(context:str, question:str):
    reader = pipeline("question-answering")
    outputs = reader(question=question, context=context)
    
    print("---" * 30, "\n", "Question Answering", "\n", "---"*10)
    print(pd.DataFrame([outputs]))
    print("---" * 30)


def summarization(text):
    summarizer = pipeline("summarization")
    outputs = summarizer(text, max_length=45, clean_up_tokenization_spaces=True)
    
    print("---" * 30, "\n", "Summarization", "\n", "---"*10)
    print(outputs[0]["summary_text"])
    print("---" * 30)


def translation(text):
    translator = pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de")
    outputs = translator(text, clean_up_tokenization_spaces=True, min_length=100)

    print("---" * 30, "\n", "Translation", "\n", "---"*10)
    print(outputs[0]["translation_text"])
    print("---" * 30)


def text_generation(prompt):
    generator = pipeline("text-generation")
    outputs = generator(prompt, max_length=200)

    print("---" * 30, "\n", "Text Generation", "\n", "---"*10)
    print(outputs[0]["generated_text"])
    print("---" * 30)



## ---------------------------------------------------------------------------------##

if __name__ == "__main__":
    text_classification(TEXT)
    named_entity_recognition(TEXT)
    ## -------------------------------------##
    question = "What does the customer want?"
    question_answering(TEXT, question)
    ## -------------------------------------##
    summarization(TEXT)
    translation(TEXT)
    ## -------------------------------------##
    response = "Dear Bumblebee, I am sorry to hear that your order was mixed up."
    prompt = TEXT + "\n\nCustomer Service Response:\n" + response
    text_generation(prompt)
    ## -------------------------------------##

## ---------------------------------------------------------------------------------##
