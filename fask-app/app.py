
from flask import Flask,render_template,request
app=Flask(__name__)
from preprocessing_output import main

import mlflow
import dagshub
import pickle


mlflow.set_tracking_uri('https://dagshub.com/Memeh15ak/British_airways_reviews.mlflow')
dagshub.init(repo_owner='Memeh15ak', repo_name='British_airways_reviews', mlflow=True)

model_name='final_british_rf'
model_version = 3

model_uri=f'models:/{model_name}/{model_version}'
model=mlflow.pyfunc.load_model(model_uri)

vectorizer=pickle.load(open('models/tfidf.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html',result=None )

@app.route('/predict',methods=['POST'])
def predict():
    text=request.form['text']
    text=main(text)
    features=vectorizer.transform([' '.join([word for word, _ in text])])
    result= model.predict(features)
    return render_template('index.html',result=result[0])

    
app.run(debug=True)