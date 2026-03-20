import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB

st.set_page_config(page_title="Review Sentiment Pro")

@st.cache_resource
def load_model():
            dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
            nltk.download('stopwords')
            ps = PorterStemmer()
            all_stopwords = stopwords.words('english')
            all_stopwords.remove('not')
            corpus = []
            for i in range(0, 1000):
                          review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
                          review = review.lower().split()
                          review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
                          corpus.append(' '.join(review))
                        cv = CountVectorizer(max_features=1500)
  X = cv.fit_transform(corpus).toarray()
  y = dataset.iloc[:, -1].values
  classifier = GaussianNB()
  classifier.fit(X, y)
  return cv, classifier, all_stopwords, ps

cv, classifier, all_stopwords, ps = load_model()

st.title("Review Sentiment Pro")
user_review = st.text_input("Enter review:")

if st.button("Analyze"):
            if user_review:
                          review = re.sub('[^a-zA-Z]', ' ', user_review).lower().split()
                          review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
                          X_test = cv.transform([' '.join(review)]).toarray()
                          prediction = classifier.predict(X_test)
                          st.write("Positive" if prediction[0] == 1 else "Negative")
                      
