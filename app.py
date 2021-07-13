import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
st.title('Sentiment Analysis')
model = joblib.load('Sentiment_Analysis')
import contractions
def con(text):
  expand = contractions.fix(text)
  return expand 

import re
def remove_sp(text):
  pattern = r'[^A-Za-z0-9\s]'
  text = re.sub(pattern,'',text)
  return text

from nltk.tokenize.toktok import ToktokTokenizer
tokenizer  = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
  tokens = tokenizer.tokenize(text)
  tokens = [token.strip() for token in tokens]
  filtered_tokens = [token for token in tokens if token not in stopword_list]
  filtered_text = ' '.join(filtered_tokens)
  return filtered_text


from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
lemmatizer = WordNetLemmatizer()
# lemmatize string
def lemmatize_word(text):
  word_tokens = word_tokenize(text)
  w = [lemmatizer.lemmatize(word, pos='v') for word in word_tokens]
  lemmas = ' '.join(w) 
  return lemmas

from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(stop_words="english")

text= st.text_input('Enter your message')

if st.button('Predict'):
  st.write('Result....')
  text = text.lower()
  text = con(text)
  text = remove_sp(text)
  text = remove_stopwords(text)
  text = lemmatize_word(text)
  y_pred = model.predict([text])
  st.write(f'Predicted Analysis is {y_pred} ')
  
