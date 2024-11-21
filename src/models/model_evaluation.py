import pandas as pd
import numpy as np
import pickle
import os
import logging
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, roc_auc_score
import json
import mlflow
import yaml
from mlflow.tracking import MlflowClient

def setup_environment():
    """Sets up environment variables and logger configuration."""
    dagshub_token = os.getenv("DAGSHUB_PAT")
    if not dagshub_token:
        raise EnvironmentError('DAGSHUB_PAT env is not set')

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "Memeh15ak"
    repo_name = "British_airways_reviews"

    # Fix for tracking URI
    mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

    # Logger setup
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

    logger.debug('Environment setup complete.')
    return logger

def read_data(path: str, model_path: str, path2: str, logger) -> pd.DataFrame:
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

        os.makedirs('./data/interim/y_pred', exist_ok=True)
        y_pred1.to_csv('./data/interim/y_pred/y_pred.csv', index=False)
        logger.info("Predictions saved successfully.")
    except Exception as e:
        logger.error(f"An error occurred while making predictions or saving data: {e}")
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

def metrics(y_test: pd.DataFrame, y_pred: pd.DataFrame, logger) -> float:
    try:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        logger.debug("Metrics calculated successfully.")
    except Exception as e:
        logger.error(f"An error occurred while calculating metrics: {e}")
        raise
    return accuracy, precision, recall, auc

def store(path: str, accuracy: float, precision: float, recall: float, auc: float, logger):
    try:
        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        with open(path, 'w') as file:
            json.dump(metrics_dict, file, indent=4)
        logger.debug(f"Metrics stored at {path}.")
    except Exception as e:
        logger.error(f"An error occurred while storing metrics: {e}")
        raise

def save_model_info(run_id:str, model_path:str, path:str, logger):
    try:
        model_info = {
            'run_id': run_id,
            'model_path': model_path
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug(f"Model info saved at {path}.")
    except Exception as e:
        logger.error(f"An error occurred while storing model info: {e}")
        raise

def main():
    logger = setup_environment()  # Set up environment and logger
    
    mlflow.set_experiment("dvc")

    with mlflow.start_run() as run:
        try:
            path = './data/interim/tfidf_test.csv'
            path2 = './data/interim/y_test.csv'
            model_path = './model.pkl'
            tfidf_test, rf_model, y_test, y_pred = read_data(path, model_path, path2, logger)
            
            accuracy, precision, recall, auc = metrics(y_test, y_pred, logger)
            mlflow.log_metric('accuracy', accuracy)
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('recall', recall)
            mlflow.log_metric('auc', auc)
            
            if hasattr(rf_model, 'get_params'):
                params = rf_model.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
            
            
            mlflow.sklearn.log_model(rf_model, "random_forest")
            save_model_info(run.info.run_id, "model", 'reports/exp_info.json',logger)
            mlflow.log_artifact('./model_info.json')

            # Log the evaluation errors log file to MLflow
            mlflow.log_artifact('model_evaluation_errors.log')
            mlflow.set_tag('author', 'mehak')
            mlflow.set_tag("experiment1", 'rf')

            metrics_path = './metrics.json'
            store(metrics_path, accuracy, precision, recall, auc, logger)
            
            mlflow.log_artifact(metrics_path)
        
        except Exception as e:
            logger.error(f"An unexpected error occurred in the main function: {e}")
            raise

if __name__ == "__main__":
    main()
