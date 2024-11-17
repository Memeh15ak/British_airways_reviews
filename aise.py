
import mlflow
import dagshub
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri('https://dagshub.com/Memeh15ak/British_airways_reviews.mlflow')
dagshub.init(repo_owner='Memeh15ak', repo_name='British_airways_reviews', mlflow=True)
experiment_name = "dvc-pipeline"
client = MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = client.create_experiment(experiment_name)
    print(f"Created new experiment with ID: {experiment_id}")
else:
    print(f"Using existing experiment with ID: {experiment.experiment_id}")

