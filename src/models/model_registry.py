import json
import mlflow
import logging
import dagshub
from mlflow.exceptions import MlflowException
import os 

dagshub_token=os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError('DAGSHUB_PAT env is not set')
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Memeh15ak"
repo_name = "British_airways_reviews"

mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

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

def register_model(model_name: str, model_info: dict):
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
        register_model(model_name, info)

    except Exception as e:
        logger.error(f"Error: An unexpected error occurred in the main function: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
