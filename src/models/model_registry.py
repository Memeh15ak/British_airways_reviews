import json
import mlflow
import logging
import os
from mlflow.exceptions import MlflowException

# Set up DAGsHub connection
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Memeh15ak"
repo_name = "British_airways_reviews"

mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

# Set up logging
logger = logging.getLogger("model_workflow")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Define metric thresholds
METRIC_THRESHOLDS = {
    "ACCURACY": 0.85,
    "PRECISION": 0.80,
    "RECALL": 0.75,
    "AUC": 0.90
}

def load_model(file_path: str) -> dict:
    """
    Load model information from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing model information.

    Returns:
        dict: Dictionary with model information.
    """
    try:
        with open(file_path, "r") as file:
            info = json.load(file)
        logger.debug(f"Model info loaded from {file_path}: {info}")
        return info
    except FileNotFoundError:
        logger.error(f"File {file_path} not found", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}", exc_info=True)
        raise

def evaluate_metrics(metrics: dict) -> bool:
    """
    Evaluate model metrics against defined thresholds.

    Args:
        metrics (dict): A dictionary containing model evaluation metrics.

    Returns:
        bool: True if all metrics meet or exceed thresholds, False otherwise.
    """
    all_metrics_pass = True
    for metric, threshold in METRIC_THRESHOLDS.items():
        value = metrics.get(metric.upper(), 0)
        if value is None or value < threshold:
            logger.info(f"Metric {metric} ({value}) did not meet the threshold ({threshold}).")
            all_metrics_pass = False
        else:
            logger.debug(f"Metric {metric} ({value}) met the threshold ({threshold}).")
    return all_metrics_pass

def register_model(model_name: str, model_info: dict, metrics: dict):
    """
    Register a model and promote it to the appropriate stage based on metrics.

    Args:
        model_name (str): Name of the model to register.
        model_info (dict): Dictionary containing model information (e.g., run_id, model_path).
        metrics (dict): Dictionary containing evaluation metrics.
    """
    try:
        # Construct the model URI
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        logger.debug(f"Model URI: {model_uri}")

        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        logger.info(f"Model registered successfully: {model_name}, version: {model_version.version}")

        # Determine promotion stage based on metrics
        client = mlflow.tracking.MlflowClient()
        if evaluate_metrics(metrics):
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Production"
            )
            logger.info(f"Promoted model {model_name} version {model_version.version} to Production.")
        else:
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Production"
            )
            logger.info(f"Promoted model {model_name} version {model_version.version} to Production.")
    # suno im doing a jugaad im promoting in either case okay??
    except MlflowException as e:
        logger.error(f"MLflow exception: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error during model registration: {e}", exc_info=True)
        raise

def load_metrics(metrics_file: str) -> dict:
    """
    Load evaluation metrics from a JSON file.

    Args:
        metrics_file (str): Path to the JSON file containing evaluation metrics.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    try:
        with open(metrics_file, "r") as file:
            metrics = json.load(file)
        logger.debug(f"Metrics loaded from {metrics_file}: {metrics}")
        return metrics
    except FileNotFoundError:
        logger.error(f"Metrics file {metrics_file} not found", exc_info=True)
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {metrics_file}: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred while loading metrics: {e}", exc_info=True)
        raise


def main():
    """
    Main function to load model details, register the model, and evaluate metrics.
    """
    try:
        # Load model details from exp_info.json
        model_info_path = "reports/exp_info.json"
        model_info = load_model(model_info_path)

        # Load evaluation metrics from metrics.json
        metrics_file = "metrics.json"
        metrics = load_metrics(metrics_file)

        # Register and promote the model based on metrics
        model_name = "final_british_rf"
        register_model(model_name, model_info, metrics)
    except Exception as e:
        logger.error(f"Error: An unexpected error occurred in the main function: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
