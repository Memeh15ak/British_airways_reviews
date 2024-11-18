import unittest
import mlflow
import os
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

class TestModelPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "Memeh15ak"
        repo_name = "British_airways_reviews"

        mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

        cls.model_name = "rf_model"
        cls.model_version = cls.get_latest_model_version(cls.model_name, stage="Staging")
        cls.model_uri = f"models:/{cls.model_name}/{cls.model_version}"
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)

        cls.test_data_path = './data/interim/tfidf_test.csv'
        cls.test_labels_path = './data/interim/y_test.csv'
        cls.test_data = pd.read_csv(cls.test_data_path)
        cls.test_labels = pd.read_csv(cls.test_labels_path).values.flatten()

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = mlflow.MlflowClient()
        latest_versions = client.get_latest_versions(model_name, stages=[stage])
        return latest_versions[0].version if latest_versions else None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.model)

    def test_model_signature(self):
        input_data = self.test_data.iloc[:1, :]  
        prediction = self.model.predict(input_data)


        self.assertEqual(input_data.shape[1], self.test_data.shape[1])

        self.assertEqual(len(prediction), 1)  

    def test_model_performance(self):
   
        predictions = self.model.predict(self.test_data)

        accuracy = accuracy_score(self.test_labels, predictions)
        precision = precision_score(self.test_labels, predictions)
        recall = recall_score(self.test_labels, predictions)
        auc = roc_auc_score(self.test_labels, predictions)

        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_auc = 0.40

        self.assertGreaterEqual(accuracy, expected_accuracy, f"Accuracy should be at least {expected_accuracy}")
        self.assertGreaterEqual(precision, expected_precision, f"Precision should be at least {expected_precision}")
        self.assertGreaterEqual(recall, expected_recall, f"Recall should be at least {expected_recall}")
        self.assertGreaterEqual(auc, expected_auc, f"AUC should be at least {expected_auc}")

if __name__ == "__main__":
    unittest.main()
