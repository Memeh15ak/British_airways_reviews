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
        # Retrieve the text input from the user
        
        # Load the pre-trained TF-IDF vectorizer
        vectorizer = pickle.load(open('models/tfidf.pkl', 'rb'))

        text = 'arshad is a good happy boy'

        # Process the text through your preprocessing function
        processed_text = main(text)

        # If `main(text)` returns a list of tuples, extract the first element of each tuple and join them into a string
        text = ' '.join([t[0] for t in processed_text])  # Assuming main returns tuples like [('word', ...)]

        # Now transform the preprocessed text
        features = vectorizer.transform([text])
        result = model.predict(features)
        print(result)
        return render_template('index.html', result=result[0])

    except Exception as e:
        # Log any errors and show an error message to the user
        print(f"Error: {e}")
        return render_template('index.html', result="An error occurred. Please try again.")

    
app.run(debug=True)
