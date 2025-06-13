import os
import nltk

# Setup a writable nltk_data directory
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download necessary NLTK data into that directory
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('movie_reviews', download_dir=nltk_data_dir)
import streamlit as st
from nltk.corpus import movie_reviews, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from wordcloud import WordCloud
import time

# Download required NLTK resources
nltk.download('movie_reviews')
nltk.download('stopwords')
nltk.download('punkt')

# Preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    return ' '.join(stemmed_tokens)

# Load dataset
@st.cache_data
def load_data():
    reviews = [(fileid, category) for category in movie_reviews.categories()
               for fileid in movie_reviews.fileids(category)]
    data, labels = [], []
    for fileid, category in reviews:
        text = movie_reviews.raw(fileid)
        processed = preprocess_text(text)
        data.append(processed)
        labels.append(category)
    return data, labels

# Train both models
@st.cache_resource
def train_models():
    data, labels = load_data()
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    nb_model = make_pipeline(CountVectorizer(), MultinomialNB())
    nb_model.fit(X_train, y_train)
    nb_acc = accuracy_score(y_test, nb_model.predict(X_test))

    lr_model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
    lr_model.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr_model.predict(X_test))

    return nb_model, nb_acc, lr_model, lr_acc

# Plot class distribution
def plot_class_distribution(labels):
    counter = Counter(labels)
    df = pd.DataFrame.from_dict(counter, orient='index').reset_index()
    df.columns = ['Sentiment', 'Count']
    fig, ax = plt.subplots()
    ax.bar(df['Sentiment'], df['Count'], color=['green', 'red'])
    ax.set_title('Sentiment Class Distribution')
    st.pyplot(fig)

# Top 10 frequent words and word cloud
def get_top_words_and_wordcloud():
    pos_words = []
    neg_words = []
    for fileid in movie_reviews.fileids('pos'):
        tokens = preprocess_text(movie_reviews.raw(fileid)).split()
        pos_words.extend(tokens)
    for fileid in movie_reviews.fileids('neg'):
        tokens = preprocess_text(movie_reviews.raw(fileid)).split()
        neg_words.extend(tokens)

    def top_n(words, n=10):
        freq = Counter(words)
        return pd.DataFrame(freq.most_common(n), columns=['Word', 'Frequency'])

    st.markdown("### 🔝 Top 10 Words in Positive Reviews")
    st.dataframe(top_n(pos_words))
    st.markdown("### 🔝 Top 10 Words in Negative Reviews")
    st.dataframe(top_n(neg_words))

    st.markdown("### 🌟 Word Cloud (Positive Reviews)")
    pos_cloud = WordCloud(width=600, height=400, background_color='white').generate(' '.join(pos_words))
    fig1, ax1 = plt.subplots()
    ax1.imshow(pos_cloud, interpolation='bilinear')
    ax1.axis("off")
    st.pyplot(fig1)

    st.markdown("### 💔 Word Cloud (Negative Reviews)")
    neg_cloud = WordCloud(width=600, height=400, background_color='white').generate(' '.join(neg_words))
    fig2, ax2 = plt.subplots()
    ax2.imshow(neg_cloud, interpolation='bilinear')
    ax2.axis("off")
    st.pyplot(fig2)

# Confidence gauge
def show_confidence_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': "Model Confidence (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "lightblue"},
                {'range': [75, 100], 'color': "lightgreen"}
            ]
        }
    ))
    st.plotly_chart(fig)

# ---- Streamlit App ----
st.set_page_config(page_title="Movie Sentiment Analyzer", layout="centered")
st.title("🎬 Movie Review Sentiment Analyzer")

st.markdown("""
This app uses **Natural Language Processing (NLP)** and classification models to analyze the sentiment of movie reviews.
""")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("- **Models**: Naive Bayes & Logistic Regression")
    st.markdown("- **Features**: CountVectorizer / TF-IDF + Porter Stemmer")
    st.markdown("- **Dataset**: NLTK Movie Reviews (1,000 samples)")
    st.markdown("---")

    st.header("📊 Options")
    nb_model, nb_acc, lr_model, lr_acc = train_models()

    if st.checkbox("✅ Show Accuracy Comparison"):
        st.info(f"Naive Bayes Accuracy: {nb_acc*100:.2f}%")
        st.info(f"Logistic Regression Accuracy: {lr_acc*100:.2f}%")

    if st.checkbox("📄 Show Class Distribution"):
        _, labels = load_data()
        plot_class_distribution(labels)

    if st.checkbox("🔠 Show Top Frequent Words & WordCloud"):
        get_top_words_and_wordcloud()

    if st.checkbox("🧐 Show Stemming Example"):
        example = "This movie had surprising twists and brilliant acting."
        tokens = word_tokenize(example.lower())
        stemmed = [PorterStemmer().stem(w) for w in tokens if w.isalpha()]
        st.write("**Original:**", example)
        st.write("**Stemmed Tokens:**", stemmed)

# Main input
st.markdown("## ✍️ Enter Your Review(s)")
user_input = st.text_area("Enter one or more reviews (each on a new line):", height=200)
selected_model = st.radio("Choose a model for prediction:", ('Naive Bayes', 'Logistic Regression'))

if st.button("🔍 Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter at least one review.")
    else:
        reviews = user_input.strip().split('\n')
        model = nb_model if selected_model == 'Naive Bayes' else lr_model
        results = []

        for idx, review in enumerate(reviews):
            processed = preprocess_text(review)
            prediction = model.predict([processed])[0]
            probabilities = model.predict_proba([processed])[0]
            confidence = max(probabilities)

            st.markdown(f"### 📄 Review {idx+1}")
            if prediction == 'pos':
                st.success("✅ Sentiment: Positive")
            else:
                st.error("🚫 Sentiment: Negative")
            st.info(f"Confidence: {confidence*100:.2f}%")
            show_confidence_gauge(confidence)

            tokens = word_tokenize(review.lower())
            filtered = [w for w in tokens if w.isalpha() and w not in stopwords.words('english')]
            stemmed = [PorterStemmer().stem(w) for w in filtered]
            mapping = {word: PorterStemmer().stem(word) for word in filtered}

            with st.expander("🔍 Preprocessing Details"):
                st.write("**Tokens:**", tokens)
                st.write("**Filtered (No Stopwords):**", filtered)
                st.write("**Stemmed:**", stemmed)
                st.markdown("**Word Mapping:**")
                st.table(mapping.items())

            results.append({
                "Review": review,
                "Processed": processed,
                "Prediction": prediction,
                "Confidence": f"{confidence*100:.2f}%"
            })

        # CSV download
        result_df = pd.DataFrame(results)
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📅 Download Results as CSV",
            data=csv,
            file_name='sentiment_analysis_results.csv',
            mime='text/csv'
        )
