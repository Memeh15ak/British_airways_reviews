from flask import Flask, render_template, request
import mlflow
import dagshub
import pickle
from preprocessing_output import main
import pandas as pd
import numpy as np
import os

dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError('DAGSHUB_PAT env is not set')
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Memeh15ak"
repo_name = "British_airways_reviews"

mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
app = Flask(__name__)

def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None

model_name = "final_british_rf"
model_version = get_latest_model_version(model_name)

model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.pyfunc.load_model(model_uri)
vectorizer = pickle.load(open('models/tfidf.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.form['text'] 
           
        print(text)
        
        processed_text = main(text)  # Assuming this function processes the text
        print(processed_text)
        
        features = vectorizer.transform([text])
        print(features)
        
        features_df = pd.DataFrame(features.toarray(), columns=vectorizer.get_feature_names_out())
        print(features_df)
        
        result = model.predict(features_df)
        print(result)

        return render_template('index.html', result=result[0])

    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('index.html', result="An error occurred. Please try again.")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
