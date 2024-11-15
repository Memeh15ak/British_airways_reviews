import pandas as pd
import numpy as np
import pickle
import os
import logging
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, roc_auc_score
from dvclive import Live
import yaml
import mlflow
import json
import dagshub

# Set timeout and retry configurations for MLflow
os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "5"
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "120"

mlflow.set_tracking_uri('https://dagshub.com/Memeh15ak/British_airways_reviews.mlflow')
dagshub.init(repo_owner='Memeh15ak', repo_name='British_airways_reviews', mlflow=True)

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

def read_data(path: str, model_path: str, path2: str) -> pd.DataFrame:
    try:
        tfidf_test = pd.read_csv(path)
    except FileNotFoundError:
        logger.error(f"Test data file not found at {path}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing the test data CSV file: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading test data: {e}")
        raise
    
    try:
        with open(model_path, 'rb') as file:
            rf_model = pickle.load(file)
    except FileNotFoundError:
        logger.error(f"Model file not found at {model_path}")
        raise
    except pickle.UnpicklingError as e:
        logger.error(f"Error loading the model: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading the model: {e}")
        raise

    try:
        y_pred = rf_model.predict(tfidf_test)
        y_pred1 = pd.DataFrame(y_pred)

        # Ensure the directory exists before saving the file
        os.makedirs('data/interim/y_pred', exist_ok=True)

        # Now save the predictions file
        y_pred1.to_csv('data/interim/y_pred/y_pred.csv', index=False)
        logger.info("Data saved successfully")
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving data: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while making predictions: {e}")
        raise

    try:
        y_test_1 = pd.read_csv(path2)
        y_test = y_test_1.values
    except FileNotFoundError:
        logger.error(f"y_test file not found at {path2}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing the y_test CSV file: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading y_test data: {e}")
        raise

    return tfidf_test, rf_model, y_test, y_pred

logger.debug('file loaded successfully')

def metrics(y_test: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    try:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
    except Exception as e:
        logger.error(f"An error occurred while calculating metrics: {e}")
        raise
    return accuracy, precision, recall, auc

logger.debug('metrics created')

def experiment_tracking(path: str, accuracy: float, precision: float, recall: float, auc: float):
    try:
        with open(path, 'r') as f:
            params = yaml.safe_load(f)
        
        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy', accuracy)
            live.log_metric('precision', precision)
            live.log_metric('recall', recall)
            live.log_metric('auc', auc)
            
            for param, value in params.items():
                for key, val in value.items():
                    live.log_param(f'{param}_{key}', val)
    except Exception as e:
        logger.error(f"An error occurred during experiment tracking: {e}")
        raise

logger.debug('experiment tracked successfully')

def store(path: str, accuracy: float, precision: float, recall: float, auc: float):
    try:
        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        with open(path, 'w') as file:
            json.dump(metrics_dict, file, indent=4)
    except Exception as e:
        logger.error(f"An error occurred while storing metrics: {e}")
        raise
logger.debug('metrics created')

def save_model_info(run_id, model_info, path):
    try:
        info = {
            'run_id': run_id,
            'model_path': model_info
        }
        with open(path, 'w') as file:
            json.dump(info, file, indent=4)
    except Exception as e:
        logger.error(f"An error occurred while storing model info: {e}")
        raise
logger.debug('model info saved')

def main():
    mlflow.set_experiment('dvc-pipeline')
    with mlflow.start_run() as run:
        try:
            path = 'data/interim/tfidf_test.csv'
            path2 = 'data/interim/y_test.csv'
            model_path = 'model.pkl'
            tfidf_test, rf_model, y_test, y_pred = read_data(path, model_path, path2)
            
            accuracy, precision, recall, auc = metrics(y_test, y_pred)
            mlflow.log_metric('accuracy', accuracy)
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('recall', recall)
            mlflow.log_metric('auc', auc)
            
            if hasattr(rf_model, 'get_params'):
                params = rf_model.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
            
            mlflow.sklearn.log_model(rf_model, "random_forest")
            save_model_info(run.info.run_id, 'model', 'reports/exp_info.json')
            mlflow.set_tag('author', 'mehak')
            mlflow.set_tag("experiment1", 'rf')

            experiment_tracking('params.yaml', accuracy, precision, recall, auc)
            metrics_path = 'metrics.json'
            store(metrics_path, accuracy, precision, recall, auc)
            
            mlflow.log_artifact(metrics_path)
        
        except Exception as e:
            logger.error(f"An unexpected error occurred in the main function: {e}")
            raise

if __name__ == "__main__":
    main()
