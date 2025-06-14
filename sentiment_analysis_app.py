import streamlit as st
import pandas as pd
import nltk
import random
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from nltk.stem import PorterStemmer

# ✅ Safe downloader for NLTK resources
def safe_nltk_download(resource_path, download_name=None):
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(download_name or resource_path.split('/')[-1])

# Ensure required resources
safe_nltk_download('tokenizers/punkt', 'punkt')
safe_nltk_download('corpora/stopwords')
safe_nltk_download('corpora/movie_reviews')

# Load dataset
def load_movie_reviews():
    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]
    random.shuffle(documents)
    return documents

# Preprocessing
stemmer = PorterStemmer()
stop_words = set(nltk.corpus.stopwords.words('english'))

def preprocess(doc):
    words = [stemmer.stem(w.lower()) for w in doc if w.isalpha() and w.lower() not in stop_words]
    return " ".join(words)

documents = load_movie_reviews()
X = [preprocess(doc) for doc, _ in documents]
y = [label for _, label in documents]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Streamlit UI
st.set_page_config(layout="wide")
st.title("🎬 Movie Review Sentiment Analyzer")

st.sidebar.header("📌 Options")
model_choice = st.sidebar.selectbox("Choose Model", ["Naive Bayes", "Logistic Regression"])
vectorizer_choice = st.sidebar.radio("Vectorizer", ["CountVectorizer", "TF-IDF"])

# Model selection
vectorizer = CountVectorizer() if vectorizer_choice == "CountVectorizer" else TfidfVectorizer()
model = MultinomialNB() if model_choice == "Naive Bayes" else LogisticRegression()

pipeline = Pipeline([
    ('vectorizer', vectorizer),
    ('classifier', model)
])

pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)

# Prediction section
st.subheader("🔍 Try Your Own Review")
user_input = st.text_area("Enter a movie review:")

if user_input:
    # ✅ This line sometimes fails if punkt is missing, so we made sure it's downloaded above
    tokens = nltk.word_tokenize(user_input)
    processed = preprocess(tokens)
    prediction = pipeline.predict([processed])[0]
    st.markdown(f"**Sentiment:** `{prediction}`")

# Accuracy
st.sidebar.markdown(f"📈 Model Accuracy: `{accuracy:.2f}`")
st.sidebar.markdown("🔹 Dataset: NLTK Movie Reviews")
st.sidebar.markdown("🔹 Preprocessing: Stopword Removal, Stemming")
