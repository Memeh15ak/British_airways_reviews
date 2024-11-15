import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

import mlflow 
import dagshub
import pickle

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

def clean(text: str) -> str:
    tex = re.sub('[^A-Za-z\s]+', '', str(text))
    return tex

def lower_case(text: str) -> str:
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def lemmatization(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stopwords(text: str) -> str:
    stop_words = set(stopwords.words('english'))
    text = text.split()
    text = [word for word in text if word not in stop_words]
    return " ".join(text)

def tokenize(text: str) -> list:
    return word_tokenize(text)

def main(text: str) -> str:
    text = clean(text)
    text = lower_case(text)
    text = remove_stopwords(text)
    text = lemmatization(text)
    tokens = tokenize(text)
    return tokens  

# model = pickle.load(open('model.pkl', 'rb'))

# vectorizer = pickle.load(open('models/tfidf.pkl', 'rb'))


# text = 'i am very happy and impressed'
        

# print(text)
  
# processed_text = main(text) 
# print(processed_text)
# features = vectorizer.transform([text])
# print(features)
# features_df = pd.DataFrame(features.toarray(), columns=vectorizer.get_feature_names_out())
# print(features_df)
# result = model.predict(features_df)
# print(result)

