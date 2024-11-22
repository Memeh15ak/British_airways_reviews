import json
import mlflow
import logging
import dagshub
from mlflow.exceptions import MlflowException
import os

# Set up environment variables
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError('DAGSHUB_PAT env is not set')
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Memeh15ak"
repo_name = "British_airways_reviews"

mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# Set up logging
logger = logging.getLogger('model_management')
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


def load_model(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            info = json.load(file)
        return info
    except FileNotFoundError:
        logger.error('File not found', exc_info=True)
        raise
    except Exception as e:
        logger.error(f'Unexpected error occurred: {e}', exc_info=True)
        raise


def validate_model_info(model_info: dict):
    required_keys = ['run_id', 'model_path']
    if not all(key in model_info for key in required_keys):
        logger.error(f"Missing keys in model_info. Required: {required_keys}, Found: {model_info.keys()}")
        raise ValueError(f"Model info must contain: {required_keys}")


def get_model_metrics(run_id: str) -> dict:
    """Fetch the metrics of the model from the MLflow run."""
    try:
        client = mlflow.tracking.MlflowClient()
        metrics = client.get_run(run_id).data.metrics
        logger.debug(f"Fetched metrics for run {run_id}: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Failed to fetch metrics for run {run_id}: {e}", exc_info=True)
        raise


def should_promote_to_production(metrics: dict, thresholds: dict) -> bool:
    """
    Determine if the model should be promoted to production based on metrics.
    Args:
        metrics (dict): Dictionary of model metrics.
        thresholds (dict): Threshold values for promotion.
    Returns:
        bool: True if the model meets the criteria for promotion, False otherwise.
    """
    for metric, threshold in thresholds.items():
        if metrics.get(metric, 0) < threshold:
            logger.info(f"Metric {metric} below threshold: {metrics.get(metric)} < {threshold}")
            return False
    return True


def register_model(model_name: str, model_info: dict, thresholds: dict):
    try:
        validate_model_info(model_info)
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        logger.debug(f"Model URI: {model_uri}")
        model_version = mlflow.register_model(model_uri, model_name)
        logger.info(f"Model registered. Version: {model_version.version}")

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage='Staging'
        )
        logger.info(f"Model {model_name} transitioned to 'Staging' stage")

        # Evaluate metrics and promote to production if thresholds are met
        metrics = get_model_metrics(model_info['run_id'])
        if should_promote_to_production(metrics, thresholds):
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage='Production'
            )
            logger.info(f"Model {model_name} transitioned to 'Production' stage")
        else:
            logger.info(f"Model {model_name} did not meet the criteria for Production.")

    except MlflowException as e:
        logger.error(f"MLflow exception: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise


def main():
    try:
        file_path = 'reports/exp_info.json'
        info = load_model(file_path)

        model_name = 'final_british_rf'
        # Define metric thresholds for promotion
        thresholds = {
            'accuracy': 0.40,
            'precision': 0.40,
            'recall': 0.40,
            'f1_score': 0.40
        }
        register_model(model_name, info, thresholds)

    except Exception as e:
        logger.error(f"Error: An unexpected error occurred in the main function: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
