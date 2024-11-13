import json
import mlflow
import logging
import dagshub


mlflow.set_tracking_uri('https://dagshub.com/Memeh15ak/British_airways_reviews.mlflow')
dagshub.init(repo_owner='Memeh15ak', repo_name='British_airways_reviews', mlflow=True)

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

def load_model(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            info = json.load(file)
        return info
    except FileNotFoundError:
        logger.error('File not found')
        raise
    except Exception as e:
        logger.error(f'Unexpected error occurred: {e}')
        raise
def register_model(model_name:str,model_info:dict):
    try:
        model_uri=f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        model_version=mlflow.register_model(model_uri,model_name)
        
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage='Staging'
        )
    except Exception as e :
        logger.error('model version registering error')
        raise
def main():
    try:
        file_path='reports/exp_info.json'
        info=load_model(file_path)
        model_name='final_british_rf'
        register_model(model_name,info)
    except Exception as e:
        logger.error(f"Error: An unexpected error occurred in the main function: {e}")
        raise

if __name__ == "__main__":
    main()
    