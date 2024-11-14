from flask import Flask, render_template, request
import mlflow
import dagshub
import pickle
from preprocessing_output import main

app = Flask(__name__)

# Set up MLflow and Dagshub integration
mlflow.set_tracking_uri('https://dagshub.com/Memeh15ak/British_airways_reviews.mlflow')
dagshub.init(repo_owner='Memeh15ak', repo_name='British_airways_reviews', mlflow=True)

# Load the registered model from MLflow
model_name = 'final_british_rf'
model_version = 11
model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.pyfunc.load_model(model_uri)

# Load the pre-trained TF-IDF vectorizer from a pickle file
vectorizer = pickle.load(open('models/tfidf.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.form['text']
        
        # Apply preprocessing to the text (assuming `main` returns preprocessed text)
        text = main(text)  # Ensure `text` is a list of words or tokens
        
        # Transform the text using the pre-trained vectorizer
        features = vectorizer.transform([text])  # Apply transform, not fit_transform
        
        # Make predictions using the MLflow model
        result = model.predict(features)
        
        # Return the prediction result to the user
        return render_template('index.html', result=result[0])
    
    except Exception as e:
        # Log any errors and show an error message to the user
        print(f"Error: {e}")
        return render_template('index.html', result="An error occurred. Please try again.")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
