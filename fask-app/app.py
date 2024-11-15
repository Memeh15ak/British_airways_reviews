from flask import Flask, render_template, request
import mlflow
import dagshub
import pickle
import logging
from preprocessing_output import main
import pandas as pd
import numpy as np


app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri('https://dagshub.com/Memeh15ak/British_airways_reviews.mlflow')
dagshub.init(repo_owner='Memeh15ak', repo_name='British_airways_reviews', mlflow=True)

model_name = 'final_british_rf'
model_version = 24
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
        
        processed_text = main(text) 
        print(processed_text)
        features = vectorizer.transform([text])
        print(features)
        features_df = pd.DataFrame(features.toarray(), columns=vectorizer.get_feature_names_out())
        print(features_df)
        result = model.predict(features_df)
        print(result)
        logger.debug(f"Prediction result: {result}")

        return render_template('index.html', result=result[0])

    except Exception as e:
    
        logger.error(f"Error during prediction: {e}")
        return render_template('index.html', result="An error occurred. Please try again.")

if __name__ == "__main__":
    app.run(debug=True)
