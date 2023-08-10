# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 19:27:17 2023

@author: Dell
"""

import streamlit as st
import pickle

def prepare_data_for_prediction(input_data):
    


    
    feature2_value = input_data['AT_K']
    feature3_value = input_data['PT_K']
    feature4_value = input_data['RS_rpm']
    feature5_value = input_data['Torque']
    feature6_value = input_data['TW_min']
    
    
    X_new = [feature2_value,feature3_value,feature4_value,feature5_value,feature6_value]

   

    return X_new

def get_prediction_label(prediction):
    # Assuming '0' corresponds to 'Not Machine Fail' and '1' corresponds to 'Machine Fail'
    return "Machine Fail" if prediction == 1 else "Not Machine Fail"

def main():
    st.title('Machine Failure Classifier Prediction App')

    # Create input fields for user input
   
    AT_K = st.text_input('Air temperature', value=' ')
    PT_K = st.text_input('Process temperature', value=' ')
    RS_rpm = st.text_input('Rotational speed', value=' ')
    Torque = st.text_input('Torque', value=' ')
    TW_min = st.text_input('Tool wear', value=' ')
    

    if st.button('Predict'):
        # Collect user inputs into a dictionary
        input_data = {
            
            'AT_K': AT_K,
            'PT_K': PT_K,
            'RS_rpm': RS_rpm,
            'Torque': Torque,
            'TW_min': TW_min
          
        }

        # Preprocess the data for prediction
        X_new = prepare_data_for_prediction(input_data)

        # Load the trained model using pickle
        with open('Machine_failure_project.pkl', 'rb') as file:
            model_rf = pickle.load(file)

        # Make predictions using the loaded model
        prediction = model_rf.predict([X_new])[0]

        # Convert the numerical prediction to text label
        prediction_label = get_prediction_label(prediction)

        st.write(f'Prediction: {prediction_label}')

if __name__ == '__main__':
    main()