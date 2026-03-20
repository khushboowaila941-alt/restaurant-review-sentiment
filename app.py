import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
import os

# --- PAGE CONFIG ---
st.set_page_config(
          page_title="Review Sentiment Pro",
          page_icon="*",
          layout="centered"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        .main {
                background-color: #f5f7f9;
                        font-family: 'Inter', sans-serif;
                            }
                                .stTextInput > div > div > input {
                                        border-radius: 10px;
                                                border: 1px solid #ddd;
                                                        padding: 15px;
                                                                font-size: 16px;
                                                                    }
                                                                        .stButton > button {
                                                                                background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
                                                                                        color: white;
                                                                                                border-radius: 20px;
                                                                                                        padding: 10px 25px;
                                                                                                                font-weight: bold;
                                                                                                                        border: none;
                                                                                                                                transition: 0.3s;
                                                                                                                                    }
                                                                                                                                        .stButton > button:hover {
                                                                                                                                                transform: scale(1.05);
                                                                                                                                                        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                                                                                                                                                            }
                                                                                                                                                                .prediction-card {
                                                                                                                                                                        padding: 20px;
                                                                                                                                                                                border-radius: 15px;
                                                                                                                                                                                        background-color: white;
                                                                                                                                                                                                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                                                                                                                                                                                                        text-align: center;
                                                                                                                                                                                                                margin-top: 20px;
                                                                                                                                                                                                                    }
                                                                                                                                                                                                                        .positive {
                                                                                                                                                                                                                                color: #27ae60;
                                                                                                                                                                                                                                        font-weight: bold;
                                                                                                                                                                                                                                                font-size: 24px;
                                                                                                                                                                                                                                                    }
                                                                                                                                                                                                                                                        .negative {
                                                                                                                                                                                                                                                                color: #e74c3c;
                                                                                                                                                                                                                                                                        font-weight: bold;
                                                                                                                                                                                                                                                                                font-size: 24px;
                                                                                                                                                                                                                                                                                    }
                                                                                                                                                                                                                                                                                        </style>
                                                                                                                                                                                                                                                                                        """, unsafe_allow_html=True)

# --- APP SETUP ---
@st.cache_resource
def load_data_and_model():
          dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

    nltk.download('stopwords')
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

    corpus = []
    for i in range(0, 1000):
                  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
                  review = review.lower()
                  review = review.split()
                  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
                  review = ' '.join(review)
                  corpus.append(review)

    cv = CountVectorizer(max_features=1500)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values

    classifier = GaussianNB()
    classifier.fit(X, y)

    return cv, classifier, all_stopwords, ps

cv, classifier, all_stopwords, ps = load_data_and_model()

# --- HEADER ---
st.title("Review Sentiment Pro")
st.markdown("Enter a restaurant review below and our AI will predict if it's **Positive** or **Negative**.")

# --- INPUT ---
user_review = st.text_input("Review text", placeholder="e.g., The food was absolutely delicious and the service was amazing!")

if st.button("Analyze Sentiment"):
          if user_review.strip() == "":
                        st.warning("Please enter a review first.")
else:
              # Preprocessing user input
              new_review = re.sub('[^a-zA-Z]', ' ', user_review)
              new_review = new_review.lower()
              new_review = new_review.split()
              new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
              new_review = ' '.join(new_review)

        # Prediction
              new_X_test = cv.transform([new_review]).toarray()
        prediction = classifier.predict(new_X_test)

        # Display Result
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        if prediction[0] == 1:
                          st.markdown('Predicted Sentiment: <span class="positive">POSITIVE</span>', unsafe_allow_html=True)
                          st.success("Glad you enjoyed it!")
else:
                  st.markdown('Predicted Sentiment: <span class="negative">NEGATIVE</span>', unsafe_allow_html=True)
                  st.error("Sorry to hear about the bad experience.")
              st.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER ---
st.divider()
st.markdown("Built with Python, Scikit-Learn, and Streamlit.")
