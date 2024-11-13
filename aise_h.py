import mlflow
import dagshub

mlflow.set_tracking_uri('https://dagshub.com/Memeh15ak/British_airways_reviews.mlflow')
dagshub.init(repo_owner='Memeh15ak', repo_name='British_airways_reviews', mlflow=True)

model_name='final_british_rf'
model_version="6"
model_uri = f"models:/{model_name}/{model_version}"

# Download model artifact to a local directory
local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)

# Load the model from the downloaded directory
model = mlflow.pyfunc.load_model(local_path)
