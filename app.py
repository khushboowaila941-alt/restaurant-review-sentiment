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
st.title("Review Sentiment Pro")

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
nltk.download('stopwords')
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
corpus = []
for i in range(1000):
              r = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]).lower().split()
              r = [ps.stem(w) for w in r if w not in set(all_stopwords)]
              corpus.append(' '.join(r))

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values
c = GaussianNB()
c.fit(X, y)

t = st.text_input("Review:")
if st.button("Analyze"):
              r = re.sub('[^a-zA-Z]', ' ', t).lower().split()
              r = [ps.stem(w) for w in r if w not in set(all_stopwords)]
              p = c.predict(cv.transform([' '.join(r)]).toarray())
              st.write("Positive" if p[0] == 1 else "Negative")
            
