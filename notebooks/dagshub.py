import mlflow
import dagshub

mlflow.set_tracking_url('https://github.com/Memeh15ak/British_airways_reviews.git')
dagshub.init(repo_owner="Mehak",repo_name="british_airways_reviews",mlflow=True)

with mlflow.start_run():
    mlflow.log_param('parameter_name','value')