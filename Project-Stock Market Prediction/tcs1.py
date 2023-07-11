# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 22:54:42 2023

@author: Dell
"""

import numpy as np
import pandas as pd
import streamlit as st
import pickle
from prophet import Prophet
from datetime import timedelta, date
Model = pickle.load(open("Prophet.pkl","rb"))

def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)

def Prediction(y,m,d):
    weekdays = [6,7]
    dates = []
    for dt in daterange(date(2013,4,5),date(y,m,d)):
        if dt.isoweekday() not in weekdays:
            dates.append(dt.strftime("%Y-%m-%d"))
    df_test = pd.DataFrame(dates)
    df_test.columns = ['ds']
    pred = Model.predict(df_test)
    data = list(pred[-1:]['ds'])
    prediction = list(pred[-1:]['yhat'])
    return data, prediction
	
def main():
    st.title("TCS Stock Price Prediction Gr-6")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style ="color:white;text-align:center;>Streamlit Bank Authenticator ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    y = st.number_input('Enter year value:',2023)
    m = st.number_input('Enter month value:',value = 4)
    d = st.number_input('Enter date value:',value = 1)
    result1= ""
    result2= ""
    if st.button("Predict"):
        result1,result2 =Prediction(y,m,d)
    st.success('The pediction interval date {}'.format(result1))
    st.success('The predicted values is {}'.format(result2))

if __name__ == '__main__':
    main()


