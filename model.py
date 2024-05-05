import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('./trained_model.sav','rb'))

#creating a function for prediction

def heart_prediction(input_data):

    try:
        input_data_numeric = [float(value) for value in input_data]
        input_numpy = np.asarray(input_data_numeric)
        
        # Reshape the input data for prediction
        input_data_reshape = input_numpy.reshape(1, -1)
        
        # Make the prediction
        prediction = loaded_model.predict(input_data_reshape)
        
        # Return prediction result
        if prediction[0] == 0:
            return 'The person is not diabetic!'
        else:
            return 'The person is diabetic!'
    except ValueError:
        return "Invalid input. Please provide numerical/floating values for all fields."
    
def main():
    
    #giving a title
    st.title('Heart-Disease-Prediction-App')
    
    #getiing input data from user 
    
    age = (st.text_input('Enter the age of person'))
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
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       