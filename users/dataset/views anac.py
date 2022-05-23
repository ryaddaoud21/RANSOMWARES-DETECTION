# -*- coding: utf-8 -*-
"""
Created on Fri May 20 15:28:14 2022

@author: User
"""

import pandas as pd
import pickle


def prediction(file):
    
    data = pd.read_csv('recap.csv')
    model = pickle.load(open("rf.sav", "rb"))
    prediction = model.predict(data)
    
    if prediction == 0:
        return "not ransomware"
    elif prediction == 1:
        return "ransomware"
    else:
        return "error"

    



    