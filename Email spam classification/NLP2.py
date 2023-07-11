"""
Created by 27-04-2023
@author: Akash Shinde
"""

import numpy as np
import pandas as pd
import re
import string
import streamlit as st
import pickle
from sklearn.linear_model import LogisticRegression
import nltk
nltk.download('stopwords')
from collections import Counter
from nltk.corpus import stopwords

cv = pickle.load(open("Spliting.pkl","rb"))
logreg = pickle.load(open("Logistic.pkl","rb"))

def Prediction(y):
    text = remove_spaces(y)
    text = clean_text(text)
    text = extraspace(text)
    text = remove_stopword(text)
    data = pd.DataFrame({'content':text},index = [0])
    test = cv.transform(data['content']).toarray()
    pred = logreg.predict(test)
    p = pred[0]
    if p == 0:
       return 'Abusive'
    if p == 1:
       return 'Not Abusive'
	
def main():
    st.title("NLP Machine Learning Problem")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style ="color:white;text-align:center;>Streamlit Bank Authenticator ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    y = st.text_input('Enter year value:','How are you?')
    result1= ""
    if st.button("Prediction"):
        result1 = Prediction(y)
    st.success('This email is {}'.format(result1))


def remove_spaces(text1):
    text1=text1.strip()
    text1=text1.split()
    return ' '.join(text1)

def clean_text(Text1):
    Text1=str(Text1).lower()
    Text1=re.sub('\[.*?\]', '',Text1)
    Text1=re.sub('https?://\S+|www\.\S+', '',Text1)
    Text1=re.sub('<.*?>+', '',Text1)
    Text1=re.sub('[%s]' % re.escape(string.punctuation), ' ',Text1)
    Text1=re.sub('\n', ' ',Text1)
    Text1=re.sub('\w*\d\w*', '',Text1)
    return Text1

def extraspace(text):
    text = " ".join(text.split())
    return text

def remove_stopword(text1):
    stop_words = stopwords.words('english')
    stopwords_dict=Counter(stop_words)
    text1=' '.join([word for word in text1.split() if word not in stopwords_dict])
    return text1


if __name__ == '__main__':
    main()



