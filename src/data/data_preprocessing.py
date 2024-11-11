import pandas as pd
import numpy as np
import nltk
from nltk.stem import SnowballStemmer
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
import os
import logging

logger=logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler=logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')

formatter=logging.Formatter('%(asctime)sss - %(name)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load(path : str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
        logger.info(f"Successfully read {path}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
    except pd.errors.ParserError as e:
        logger.error(f"Parsing error while reading {path}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading the file: {e}")
        
logger.debug('dataframe sucessfully loaded')
        

def clean(text : str) -> str:
    try:
        tex = re.sub('[^A-Za-z]+', ' ', str(text))
        return tex
    except Exception as e:
        logger.error(f"An error occurred during cleaning text: {e}")
        return text
logger.debug('data cleaned succesfully')

def tokenize_words(text: str) -> str:
    try:
        sent_tokenized = [sent for sent in nltk.sent_tokenize(text)]
        words_tokenized = [word for word in nltk.word_tokenize(sent_tokenized[0])]
        filtered = [word for word in words_tokenized if re.search('[a-zA-Z]', word)]

        stemmer = SnowballStemmer('english')
        stems = [stemmer.stem(word) for word in filtered]

        stop_words = set(stopwords.words('english'))
        words_filtered = [word for word in stems if word.lower() not in stop_words]
        return words_filtered
    except Exception as e:
        logger.error(f"An error occurred during tokenization: {e}")
        return []
logger.debug('tokenization and stemming applied sucessfully')

def pos(text : str) -> str:
    try:
        tagged = nltk.pos_tag(text)
        return tagged
    except Exception as e:
        logger.error(f"An error occurred during POS tagging: {e}")
        return []
logger.debug('parts of speech tagging applied sucessfully')

def func(text: str) -> str:
    try:
        sia = SentimentIntensityAnalyzer()
        score = sia.polarity_scores(text)
        return "positive" if score['compound'] > 0 else "negative"
    except Exception as e:
        logger.error(f"An error occurred during sentiment analysis: {e}")
        return "neutral"
logger.debug('sentiment analysis done sucessfully')

def store_data(path :str, file_n:str, df:pd.DataFrame):
    try:
        os.makedirs(path, exist_ok=True)
        file_name = file_n
        full_path = os.path.join(path, file_name)
        df.to_csv(full_path, index=False)
        print(f"Data saved to {full_path}")
    except Exception as e:
        logger.error(f"An error occurred while saving data to {path}: {e}")
logger.debug('data stored sucessfully ')

def main():
    try:
        path = 'data\\raw\\raw_data.csv'
        df = load(path)
        
        if df is not None:
            df['reviews'] = df['reviews'].apply(clean)
            df['reviews_words'] = df['reviews'].apply(tokenize_words)
            df['pos'] = df['reviews_words'].apply(pos)
            df['sentiment'] = df['reviews'].apply(func)

            data_path = 'data/processed'
            file_n = 'processed_data.csv'
            df_new = df[['reviews', 'sentiment']]
            store_data(data_path, file_n, df_new)
        else:
            logger.error("Data could not be loaded. Please check the file path and try again.")
    
    except Exception as e:
        logger.error(f"An unexpected error occurred in the main function: {e}")

if __name__ == "__main__":
    main()
