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


def register_model(model_name: str, model_info: dict):
    try:
        # Construct the model URI
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        
        # Log model URI for debugging
        logger.debug(f"Model URI: {model_uri}")

        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        logger.debug(f"Registered model version: {model_version.version}")

        # Transition model to Production stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage='Production'
        )
        logger.info(f"Model {model_name} transitioned to 'Production' stage")

    except MlflowException as e:
        logger.error(f"MLflow exception: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error during model registration: {e}", exc_info=True)
        raise


def main():
    try:
        file_path = './reports/exp_info.json'
        info = load_model(file_path)
        
        model_name = 'final_british_rf'
        register_model(model_name, info)

    except Exception as e:
        logger.error(f"Error: An unexpected error occurred in the main function: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
