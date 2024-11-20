# promote model

import os
import mlflow

def promote_model():
    
        dagshub_token=os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError('DAGSHUB_PAT env is not set')
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        dagshub_url = "https://dagshub.com"
        repo_owner = "Memeh15ak"
        repo_name = "British_airways_reviews"
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
        
        client = mlflow.MlflowClient()
        model_name = "final_british_rf"
    # Get the latest version in staging
        latest_version_staging = client.get_latest_versions(model_name, stages=["Staging"])[0].version

    # Archive the current production model
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        for version in prod_versions:
            client.transition_model_version_stage(
                name=model_name,
                version=version.version,
                stage="Archived"
            )

        # Promote the new model to production
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version_staging,
            stage="Production"
    )
        print(f"Model version {latest_version_staging} promoted to Production")

if __name__ == "__main__":
    promote_model()