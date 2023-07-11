# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 16:42:48 2023

@author: Dell
"""


import numpy as np
import pandas as pd
import streamlit as st
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta, date
Model = pickle.load(open("ARIMA.pkl","rb"))

def Prediction(input_date):
    pred = Model.predict(start='2013-04-05',end=input_date)
    x = pred.index[-1]
    y = pred.values[-1]
    return x,y
	
def main():
    st.title("Infosys Stock Price Prediction Gr-6")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style ="color:white;text-align:center;>Streamlit Bank Authenticator ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    z = st.text_input("Date","Type here in this format(e.g:2023-04-05)")
    result1=""
    result2= ""
    if st.button("Predict"):
        result1,result2 =Prediction(z)
    st.success('The pediction date {}'.format(result1))
    st.success('The predicted stock value is {}'.format(result2))

if __name__ == '__main__':
    main()

    
