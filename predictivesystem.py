# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 21:37:36 2024

@author: jatin
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle
loaded_model = pickle.load(open('./trained_model.sav','rb'))

input = (58,1,0,146,218,0,1,105,0,2,1,1,3)

convert_numpy = np.asarray(input)

input_reshape = convert_numpy.reshape(1,-1)

prediction = loaded_model.predict(input_reshape)

if ((prediction[0])==1):
    print("Person have Heart Disease !")
else:
    print("Person is free from heart disease !")