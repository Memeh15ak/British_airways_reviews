FROM python:3.11.5

WORKDIR /app

COPY flask_app/ /app/

COPY models/tfidf.pkl /app/models/tfidf.pkl

RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet punkt averaged_perceptron_tagger maxent_ne_chunker words

EXPOSE 5000

CMD ["python","app.py"]
