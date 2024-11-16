import pandas as pd
import numpy as np
import pickle
import os
import logging
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, roc_auc_score
import yaml
import mlflow
import json

# Setting up environment variables
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError('DAGSHUB_PAT env is not set')
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Memehak15ak"
repo_name = "British_airways_reviews"

mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# Logger setup
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Read data and model
def read_data(path: str, model_path: str, path2: str):
    try:
        tfidf_test = pd.read_csv(path)
        logger.info(f"Successfully read test data from {path}")
    except Exception as e:
        logger.error(f"Failed to read test data from {path}: {e}")
        raise

    try:
        with open(model_path, 'rb') as file:
            rf_model = pickle.load(file)
        logger.info(f"Successfully loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise

    try:
        y_pred = rf_model.predict(tfidf_test)
        y_pred1 = pd.DataFrame(y_pred)
        os.makedirs('data/interim/y_pred', exist_ok=True)
        y_pred1.to_csv('data/interim/y_pred/y_pred.csv', index=False)
        logger.info("Predictions saved successfully.")
    except Exception as e:
        logger.error(f"Failed to make predictions or save data: {e}")
        raise

    try:
        y_test_1 = pd.read_csv(path2)
        y_test = y_test_1.values
        logger.info(f"Successfully read y_test data from {path2}")
    except Exception as e:
        logger.error(f"Failed to read y_test data from {path2}: {e}")
        raise

    return tfidf_test, rf_model, y_test, y_pred

# Metrics function
def metrics(y_test, y_pred):
    try:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        logger.info(f"Metrics calculated: accuracy={accuracy}, precision={precision}, recall={recall}, auc={auc}")
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        raise
    return accuracy, precision, recall, auc

# Experiment tracking using MLflow
def experiment_tracking(path: str, accuracy: float, precision: float, recall: float, auc: float):
    try:
        with open(path, 'r') as f:
            params = yaml.safe_load(f)
        
        # Use MLflow for experiment tracking
        with mlflow.start_run() as run:
            mlflow.log_metric('accuracy', accuracy)
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('recall', recall)
            mlflow.log_metric('auc', auc)
            logger.info("Logged metrics to MLflow.")
            
            for param, value in params.items():
                for key, val in value.items():
                    mlflow.log_param(f'{param}_{key}', val)
            logger.info("Logged parameters to MLflow.")
    except Exception as e:
        logger.error(f"Error during experiment tracking with MLflow: {e}")
        raise

# Save metrics as a JSON file
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
        logger.info(f"Metrics saved to {path}")
    except Exception as e:
        logger.error(f"Error saving metrics to {path}: {e}")
        raise

# Save model info
def save_model_info(run_id, model_info, path):
    try:
        info = {
            'run_id': run_id,
            'model_path': model_info
        }
        with open(path, 'w') as file:
            json.dump(info, file, indent=4)
        logger.info(f"Model information saved to {path}")
    except Exception as e:
        logger.error(f"Error saving model info to {path}: {e}")
        raise

# Main experiment flow
def main():
    mlflow.set_experiment('dvc-pipeline')
    with mlflow.start_run() as run:
        try:
            path = './data/interim/tfidf_test.csv'
            path2 = './data/interim/y_test.csv'
            model_path = './model.pkl'
            tfidf_test, rf_model, y_test, y_pred = read_data(path, model_path, path2)
            
            accuracy, precision, recall, auc = metrics(y_test, y_pred)
            
            # Log metrics to MLflow
            mlflow.log_metric('accuracy', accuracy)
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('recall', recall)
            mlflow.log_metric('auc', auc)
            logger.info("Metrics logged to MLflow.")
            
            # Log parameters to MLflow
            if hasattr(rf_model, 'get_params'):
                params = rf_model.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(rf_model, "random_forest")
            
            # Save model info to file
            save_model_info(run.info.run_id, 'model', 'reports/exp_info.json')

            # Log the experiment details to MLflow
            experiment_tracking('./params.yaml', accuracy, precision, recall, auc)

            # Store metrics as JSON
            metrics_path = './metrics.json'
            store(metrics_path, accuracy, precision, recall, auc)
            
            # Log the metrics file as artifact
            mlflow.log_artifact(metrics_path)

        except Exception as e:
            logger.error(f"Error in the main function: {e}")
            raise

# Entry point
if __name__ == "__main__":
    main()
