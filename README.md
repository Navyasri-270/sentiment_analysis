Movie Review Sentiment Analyzer
A web application leveraging Natural Language Processing (NLP) and machine learning to perform sentiment analysis on movie reviews. Built with Streamlit, it supports two classification models—Naive Bayes and Logistic Regression—and provides insightful visualizations to aid interpretability.

Overview
This application enables users to input movie reviews and receive sentiment classifications as either positive or negative. The sentiment models are trained on the well-known NLTK Movie Reviews dataset using robust preprocessing techniques such as tokenization, stopword removal, and stemming. Users can select between Naive Bayes and Logistic Regression models for predictions, with confidence scores presented alongside results.

Features
Dual Model Support:
Naive Bayes classifier with CountVectorizer
Logistic Regression classifier with TF-IDF vectorization

Comprehensive Text Preprocessing:
Tokenization
Stopword filtering
Porter stemming

Visual Analytics:

Sentiment class distribution bar chart
Top 10 frequent words for positive and negative reviews
Word clouds illustrating prominent terms by sentiment

Confidence gauges indicating prediction certainty

Detailed Review Analysis:

Token-level and stemming information for user inputs

Export Capability:

Downloadable CSV reports of sentiment predictions and confidence scores

Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/movie-review-sentiment-analyzer.git
cd movie-review-sentiment-analyzer
(Optional) Create and activate a virtual environment:

bash

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
Install dependencies:

bash

pip install -r requirements.txt

Run the Streamlit application:
bash

streamlit run app.py

Usage
Enter one or multiple movie reviews (each on a new line) in the provided text box.
Choose the classification model (Naive Bayes or Logistic Regression).
Click Analyze Sentiment to obtain sentiment predictions with associated confidence metrics.
Review detailed preprocessing steps via the expandable panels.
Download results as CSV for further analysis or record keeping.

Technical Details
Dataset: NLTK Movie Reviews corpus (1,000 labeled samples)
Text Processing: Lowercasing, tokenization, stopword removal, Porter stemming
Model Pipelines: Implemented via scikit-learn's Pipeline for streamlined training and inference
Visualization: Combination of Matplotlib and Plotly for static and interactive charts
Performance Optimization: Leveraging Streamlit’s caching decorators to minimize redundant computation

Dependencies
Python 3.7 or higher
Streamlit
NLTK
scikit-learn
pandas
matplotlib
plotly
wordcloud
License
This project is released under the MIT License. Contributions and adaptations are welcome.

Contact
For inquiries or feedback, please contact [Your Name] at [your.email@example.com] or visit the GitHub repository [github.com/yourusername/movie-review-sentiment-analyzer].

If you require a requirements.txt file or additional documentation, please let me know.




