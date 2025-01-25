from flask import Flask, render_template, request
import mlflow
import pickle
import pandas as pd
import numpy as np
import os
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

def clean(text: str) -> str:
    tex = re.sub('[^A-Za-z\s]+', '', str(text))
    return tex

def lower_case(text: str) -> str:
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def lemmatization(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stopwords(text: str) -> str:
    stop_words = set(stopwords.words('english'))
    text = text.split()
    text = [word for word in text if word not in stop_words]
    return " ".join(text)

def tokenize(text: str) -> list:
    return word_tokenize(text)

def normalize(text: str) -> str:
    text = clean(text)
    text = lower_case(text)
    text = remove_stopwords(text)
    text = lemmatization(text)
    tokens = tokenize(text)
    return tokens

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
        print(latest_version)
    return latest_version[0].version if latest_version else None

model_name = "final_british_rf"
model_version = get_latest_model_version(model_name)

model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.pyfunc.load_model(model_uri)
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('models/tfidf.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Received request...")
        text = request.form.get('text', '')
        print(f"Receivedd text: {text}")
        
        processed_text = normalize(text)
        print(f"Processed text: {processed_text}")
        
        features = vectorizer.transform([text])
        print(f"Transformed features: {features}")
        
        features_df = pd.DataFrame(features.toarray(), columns=vectorizer.get_feature_names_out())
        print(f"Features DataFrame: {features_df}")
        
        result = model.predict(features_df)
        print(f"Prediction result: {result}")
        
        return render_template('index.html', result=result[0])
    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('index.html', result="An error occurred. Please try again.")



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
