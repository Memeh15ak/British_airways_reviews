import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import os
import yaml
import logging
import pickle
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download("averaged_perceptron_tagger")

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(path: str) -> pd.DataFrame:
    try:
        df_new = pd.read_csv(path)
        logger.info("File loaded successfully")
        return df_new
    except FileNotFoundError:
        logger.error(f"File not found at {path}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error while parsing the CSV file: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading data: {e}")
        raise

def yaml_params(path: str) -> dict:
    try:
        with open(path, 'r') as file:
            params = yaml.safe_load(file)['feature_engineering']
        
        max_df = params['max_df']
        max_features = params['max_features']
        min_df = params['min_df']
        test_size = float(params['test_size'])  # Ensure it's a float
        random_state = params['random_state']
        
        # If 'stop_words' is 'english', use the built-in stopwords, else use custom stop words list
        stop_words = params['stop_words'] if params['stop_words'] != 'english' else stopwords.words('english')
        
        use_idf = params['use_idf']
        
        return max_df, max_features, min_df, test_size, random_state, stop_words, use_idf
    except FileNotFoundError:
        logger.error(f"YAML file not found at {path}")
        raise
    except KeyError as e:
        logger.error(f"Missing key in the YAML file: {e}")
        raise 
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading YAML file: {e}")
        raise

def test_split(test_size: float, df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            df['reviews'], df['sentiment'], test_size=test_size, random_state=random_state
        )
        logger.info("Train-test split successful")
        return X_train, X_test, y_train, y_test
    except KeyError as e:
        logger.error(f"The dataframe does not contain the expected columns: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during the train-test split: {e}")
        raise

def tfidf(max_df: float, max_features: int, min_df: float, stop_words: list, use_idf: bool, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    try:
        tfidf_vectorizer = TfidfVectorizer(
            max_df=max_df, max_features=max_features, min_df=min_df,
            stop_words=stop_words, use_idf=use_idf
        )
        tfidf_matrix_train = tfidf_vectorizer.fit_transform(X_train)
        tfidf_matrix_test = tfidf_vectorizer.transform(X_test)

        feature_names = tfidf_vectorizer.get_feature_names_out()
        tfidf_df_train = pd.DataFrame(tfidf_matrix_train.toarray(), columns=feature_names)
        tfidf_df_test = pd.DataFrame(tfidf_matrix_test.toarray(), columns=feature_names)
        tfidf_df_train['output'] = y_train.values

        label_encoder = LabelEncoder()
        tfidf_df_train['output'] = label_encoder.fit_transform(tfidf_df_train['output'])
        y_test = label_encoder.transform(y_test)
        y_test = pd.DataFrame(y_test, columns=['output'])

        pickle.dump(tfidf_vectorizer, open('models/tfidf.pkl', 'wb'))
        logger.info("TF-IDF vectorization and model saving successful")

        return tfidf_df_train, tfidf_df_test, y_test
    except ValueError as e:
        logger.error(f"Error with TF-IDF vectorization parameters: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during TF-IDF vectorization: {e}")
        raise

def store_data(path: str, df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame):
    try:
        os.makedirs(path, exist_ok=True)
        
        df1.to_csv(os.path.join(path, 'tfidf_train.csv'), index=False)
        df2.to_csv(os.path.join(path, 'tfidf_test.csv'), index=False)
        df3.to_csv(os.path.join(path, 'y_test.csv'), index=False)
        logger.info("Data saved successfully")
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving data: {e}")
        raise

def main():
    try:
        path = './data/processed/processed_data.csv'
        df_new = load_data(path)
        
        new_path = 'params.yaml'
        max_df, max_features, min_df, test_size, random_state, stop_words, use_idf = yaml_params(new_path)
        
        X_train, X_test, y_train, y_test = test_split(test_size, df_new, random_state)
        
        tfidf_df_train, tfidf_df_test, y_test = tfidf(
            max_df, max_features, min_df, stop_words, use_idf, X_train, X_test, y_train, y_test
        )
        
        data_path = './data/interim'
        store_data(data_path, tfidf_df_train, tfidf_df_test, y_test)
    
    except Exception as e:
        logger.error(f"An unexpected error occurred in the main function: {e}")

if __name__ == "__main__":
    main()
