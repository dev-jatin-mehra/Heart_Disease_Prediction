# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 21:37:02 2024

@author: jatin
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 20:44:08 2024

@author: jatin
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('./trained_model.sav','rb'))

#creating a function for prediction

def heart_prediction(input_data):

    input_data_numeric = [float(value) for value in input_data]

    input_numpy=np.asarray(input_data_numeric)
    
    #predicting for one instance
    input_data_reshape=input_numpy.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshape)

    if(prediction[0]==0):
        return 'Congratulations ! You are free from heart disease !'
    else:
        return 'I would prefer you should consult your doctor !'
    
def main():
    
    #giving a title
    st.title('Heart-Disease-Prediction-App')
    
    #getiing input data from user 
    
    age = st.text_input('Enter the age of person')
    sex = st.text_input('Enter the gender')
    cp = st.text_input('Cp value')
    trestbps = st.text_input('trestbps value')
    chol = st.text_input('Cholestrol Value')
    fbs = st.text_input('FBS value')
    restecg = st.text_input('RestEcg Value')
    thalach = st.text_input('Thalach Value')
    exang = st.text_input('Exang value')
    oldpeak = st.text_input('Oldpeak value')
    slope = st.text_input('Slope Value')
    ca = st.text_input('Ca Value')
    thal = st.text_input('Thal value')
    
    #code for prediction 
    diagnosis = ""
    
    #creating a button
    
    if st.button('GENERATE RESULT'):
       diagnosis = heart_prediction([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]) 
       
    st.success(diagnosis)
    
    
    
if __name__=='__main__':
    main()
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       