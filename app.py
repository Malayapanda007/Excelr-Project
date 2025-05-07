from flask import Flask, render_template, request
import pickle
import re
import html
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')


with open('sentiment_pipeline.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)


label_map = {
    0: 'Depression',
    1: 'Diabetes, Type 2',
    2: 'High Blood Pressure'
}


stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
punctuations = string.punctuation

def preprocess_review(text):
    text = text.lower()
    text = re.sub(r"<[^>]*>", "", text)
    text = html.unescape(text)
    text = text.strip('"').strip("'")
    for char in punctuations:
        text = text.replace(char, '')
    text = " ".join([word for word in text.split() if word not in stop_words])
    tokens = word_tokenize(text)
    stemmed = [stemmer.stem(word) for word in tokens]
    return " ".join(stemmed)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        cleaned_review = preprocess_review(review)
        pred_encoded = model.predict([cleaned_review])[0]
        pred_label = label_map.get(pred_encoded, "Unknown Condition")
        return render_template('index.html', prediction=pred_label)

if __name__ == '__main__':
    app.run(debug=True)
