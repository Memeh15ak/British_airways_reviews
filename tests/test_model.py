import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import time


class TestModelLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up environment variables
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("Environment variable DAGSHUB_PAT is not set. Please set it for authentication.")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        dagshub_url = "https://dagshub.com"
        repo_owner = "Memeh15ak"
        repo_name = "British_airways_reviews"
        mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

        # Initialize variables
        cls.new_model_name = "final_british_rf"
        cls.new_model = None

        # Retry logic for loading the model
        MAX_RETRIES = 3
        RETRY_DELAY = 10  # seconds
        for attempt in range(MAX_RETRIES):
            try:
                cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
                if cls.new_model_version is None:
                    raise ValueError(f"No model found in 'Staging' stage for {cls.new_model_name}.")

                cls.new_model_uri = f"models:/{cls.new_model_name}/{cls.new_model_version}"
                cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)
                print(f"Model loaded successfully: {cls.new_model_name}, version: {cls.new_model_version}")
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"Attempt {attempt + 1}/{MAX_RETRIES} to load model failed: {e}. Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    raise RuntimeError("Failed to load the model after multiple attempts.") from e

        # Load TF-IDF vectorizer
        vectorizer_path = "./models/tfidf.pkl"
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"TF-IDF vectorizer file {vectorizer_path} is missing.")
        with open(vectorizer_path, "rb") as f:
            cls.vectorizer = pickle.load(f)

        # Load holdout test data
        data_path = "./data/interim/tfidf_test.csv"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Holdout test data {data_path} is missing.")
        cls.holdout_data = pd.read_csv(data_path)
        if cls.holdout_data.empty:
            raise ValueError("Holdout dataset is empty.")

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = mlflow.MlflowClient()
        try:
            print(f"Fetching all versions for model: {model_name}")
            versions = client.search_model_versions(f"name='{model_name}'")
            
            staging_versions = [v for v in versions if v.current_stage == stage]
            
            if staging_versions:
                latest_version = max(staging_versions, key=lambda v: int(v.version))
                print(f"Latest version in stage {stage}: {latest_version.version}")
                return latest_version.version
            else:
                print(f"No versions found in stage {stage} for model: {model_name}")
                return None
        except Exception as e:
            print(f"Error while fetching model versions: {e}")
            return None

    def align_features(self, X):
        """Align features of the test data with those expected by the model."""
        vectorizer_features = self.vectorizer.get_feature_names_out()
        model_features = getattr(self.new_model, "input_schema", vectorizer_features)
        
        # Ensure all model features exist in input
        for missing_feature in set(model_features) - set(X.columns):
            X[missing_feature] = 0
        return X[model_features]

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model, "Model should be loaded successfully.")

    def test_model_signature(self):
        input_text = "I am so happy and impressed by this flight . The quality is good . Overall great experience"
        input_data = self.vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray(), columns=self.vectorizer.get_feature_names_out())

        # Align input features
        input_df = self.align_features(input_df)

        prediction = self.new_model.predict(input_df)
        self.assertEqual(len(prediction), input_df.shape[0])
        self.assertEqual(len(prediction.shape), 1, "Prediction output should be 1-dimensional.")

    def test_model_performance(self):
        X_holdout = self.holdout_data.iloc[:, 0:-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        X_holdout_aligned = self.align_features(X_holdout)
        y_pred_new = self.new_model.predict(X_holdout_aligned)

        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new, zero_division=1)
        recall_new = recall_score(y_holdout, y_pred_new, zero_division=1)
        f1_new = f1_score(y_holdout, y_pred_new, zero_division=1)

        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40

        self.assertGreaterEqual(accuracy_new, expected_accuracy, f"Accuracy should be at least {expected_accuracy}")
        self.assertGreaterEqual(precision_new, expected_precision, f"Precision should be at least {expected_precision}")
        self.assertGreaterEqual(recall_new, expected_recall, f"Recall should be at least {expected_recall}")
        self.assertGreaterEqual(f1_new, expected_f1, f"F1 score should be at least {expected_f1}")


if __name__ == "__main__":
    unittest.main()
