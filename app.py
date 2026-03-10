import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

medical_df = pd.read_csv('insurance.csv')

medical_df.replace({'sex':{'male':0,'female':1}},inplace=True)
medical_df.replace({'smoker':{'yes':0,'no':1}},inplace=True)
medical_df.replace({'region':{'southeast':0,'southwest':1, 'northwest':2, 'northeast':3}},inplace=True)

X=medical_df.drop('charges',axis=1)
y= medical_df['charges']
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.1, random_state=2)

scaler = StandardScaler()
scaler.fit_transform(X_train)

rfr=RandomForestRegressor()
rfr.fit(X_train, y_train)
y_pred= rfr.predict(X_test)
score= r2_score(y_test,y_pred)


#web app
st.title("MEDICAL INSURANCE PREDICTION MODEL")
input_text = st.text_input("Enter person all features")
input_text_splited = input_text.split(",")
try:
    np_df = np.asarray(input_text_splited, dtype=float)
    prediction =  rfr.predict(np_df.reshape(1,-1))
    st.write("Medical Insurance for this person is :\n",prediction[0])
except ValueError:
    st.write("Please Enter numerical value")