import pandas as pd
import numpy as np
import nltk
from nltk.stem import SnowballStemmer,WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.chunk import RegexpParser
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
nltk.download('vader_lexicon')
import re 

def clean(text: str) -> str:
    tex = re.sub('[^A-Za-z\s]+', '', str(text))
    return tex

def lower_case(text:str)->str:
    text=text.split()
    text=[word.lower() for word in text]
    return " ".join(text)      

def lemmetization(text):
    lemmatizer=WordNetLemmatizer()
    text=text.split()
    text=[lemmatizer.lemmatize(word)for word in text]
    return " ".join(text)    
  
def tokenize_words(text: str) -> str:
        sent_tokenized = [sent for sent in nltk.sent_tokenize(text)]
        words_tokenized = [word for word in nltk.word_tokenize(sent_tokenized[0])]
        filtered = [word for word in words_tokenized if re.search('[a-zA-Z]', word)]

        stemmer = SnowballStemmer('english')
        stems = [stemmer.stem(word) for word in filtered]

        stop_words = set(stopwords.words('english'))
        words_filtered = [word for word in stems if word.lower() not in stop_words]
        return words_filtered


def pos(text : str) -> str:
        tagged = nltk.pos_tag(text)
        return tagged


def main(text):
    text=clean(text)
    text=lower_case(text)
    text=lemmetization(text)
    text=tokenize_words(text)
    text=pos(text)
    
    return text
    

