from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import pickle
import yaml
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

def yaml_params(path:str)-> float:
    try:
        with open(path, 'r') as file:
            params = yaml.safe_load(file)['model_training']
            n_estimators=params['n_estimators']
            learning_rate=params['learning_rate']
            max_depth=params['max_depth']
        return n_estimators, learning_rate,max_depth
    except FileNotFoundError:
        logger.error(f"YAML file not found at {path}")
        raise
    except KeyError as e:
        logger.errorKeyError(f"Missing key in the YAML file: {e}")
    except yaml.YAMLError as e:
        logger.error (f"Error parsing YAML file: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading YAML file: {e}")

def read_data(path:str, path2:str)->pd.DataFrame:
    try:
        tfidf_df_train = pd.read_csv(path)
        tfidf_df_test = pd.read_csv(path2)
        return tfidf_df_train, tfidf_df_test
    except FileNotFoundError:
        logger.error(f"File not found at {path} or {path2}")
        raise
    except pd.errors.ParserError as e:
        logger.error (f"Error while parsing the CSV file: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading data: {e}")
        raise
    
logger.debug('')

def train(n_estimators:float, train:pd.DataFrame, learning_rate:float, max_depth:float)-> GradientBoostingClassifier:
    try:
        gbm_model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
        gbm_model.fit(train.iloc[:, 0:-1], train['output'])
        return gbm_model
    except ValueError as e:
        logger.error (f"Error in training the model: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during model training: {e}")
        raise
    
logger.debug('model trained sucessfully')


def store(path:str, gbm_model:GradientBoostingClassifier):
    try:
        with open(path, 'wb') as file:
            pickle.dump(gbm_model, file)
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving the model: {e}")
        raise
    
logger.debug("file stored sucessfully")

def main():
    try:
        n_estimators,learning_rate,max_depth = yaml_params('params.yaml')
        
        train_path = 'data/interim/tfidf_train.csv'
        test_path = 'data/interim/tfidf_test.csv'
        tfidf_df_train, tfidf_df_test = read_data(train_path, test_path)
        
        gbm_model = train(n_estimators, tfidf_df_train,learning_rate,max_depth)
        
        model_path = 'model.pkl'
        store(model_path, gbm_model)
    
    except Exception as e:
        logger.error(f"An unexpected error occurred in the main function: {e}")
        raise
if __name__ == "__main__":
    main()