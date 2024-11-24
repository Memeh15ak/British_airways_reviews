import json
import mlflow
import logging
import os
from mlflow.exceptions import MlflowException


# Setup logging
logger = logging.getLogger('model_workflow')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# Environment and DagsHub configuration
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    logger.error("DAGSHUB_PAT environment variable is not set. Exiting.")
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set.")
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Memeh15ak"
repo_name = "British_airways_reviews"

mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')


# Thresholds for model promotion
METRIC_THRESHOLDS = {
    "accuracy": 0.55,
    "precision": 0.50,
    "recall": 0.50,
    "auc": 0.50
}


def load_model(file_path: str) -> dict:
    """
    Loads the model information from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing model information.

    Returns:
        dict: Model information loaded from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: For any unexpected error.
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r') as file:
            info = json.load(file)
        logger.debug(f"Model information loaded from {file_path}")
        return info
    except FileNotFoundError as e:
        logger.error(e, exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred while loading the model file: {e}", exc_info=True)
        raise


def evaluate_metrics(metrics: dict) -> bool:
    """
    Evaluates if the model's metrics meet the required thresholds.

    Args:
        metrics (dict): A dictionary containing model evaluation metrics.

    Returns:
        bool: True if all metrics meet the thresholds, False otherwise.
    """
    for metric, threshold in METRIC_THRESHOLDS.items():
        if metrics.get(metric, 0) < threshold:
            logger.info(f"Metric {metric} did not meet the threshold. "
                        f"Value: {metrics.get(metric)}, Threshold: {threshold}")
            return False
    logger.debug("All metrics met the threshold criteria.")
    return True


def register_model(model_name: str, model_info: dict):
    """
    Registers a model in MLflow and transitions its stage based on evaluation metrics.

    Args:
        model_name (str): Name of the model to register.
        model_info (dict): Dictionary containing model details such as run ID and metrics.

    Raises:
        MlflowException: If MLflow encounters an issue during registration.
        Exception: For any unexpected error during the registration process.
    """
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        logger.debug(f"Registering model '{model_name}' with URI: {model_uri}")

        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        logger.info(f"Model registered successfully: {model_name}, version: {model_version.version}")

        # Evaluate metrics and transition to the appropriate stage
        if evaluate_metrics(model_info.get('metrics', {})):
            stage = 'Production'
            logger.info(f"Promoting model {model_name} version {model_version.version} to Production.")
        else:
            stage = 'Staging'
            logger.info(f"Promoting model {model_name} version {model_version.version} to Staging.")

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage=stage
        )
        logger.info(f"Model {model_name} version {model_version.version} transitioned to '{stage}' stage.")

    except MlflowException as e:
        logger.error(f"MLflow exception during model registration: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error during model registration: {e}", exc_info=True)
        raise


def main():
    """
    Main function to load model details, register the model, and evaluate metrics.
    """
    try:
        file_path = 'reports/exp_info.json'
        info = load_model(file_path)
        model_name = 'final_british_rf'
        register_model(model_name, info)
    except Exception as e:
        logger.error(f"Error: An unexpected error occurred in the main function: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
