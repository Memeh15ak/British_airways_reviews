from flask import Flask,render_template,request
app=Flask(__name__)
from preprocessing_output import main

import mlflow
import dagshub

mlflow.set_tracking_uri('https://dagshub.com/Memeh15ak/British_airways_reviews.mlflow')
dagshub.init(repo_owner='Memeh15ak', repo_name='British_airways_reviews', mlflow=True)

model_name='final_british_rf'
model_version=3

model_uri=f'models:/{model_name}/{model_version}'
model=mlflow.pyfunc.load_model(model_uri)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',method=['POST'])
def predict():
    text=request.form['text']
    text=main(text)
    return text 
app.run(debug=True)
    
    
