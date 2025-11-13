import pandas as pd
import streamlit as st 
from sklearn.linear_model import LogisticRegression

st.title('Model Deployment: Logistic Regression')

st.sidebar.header('User Input Parameters')

def user_input_features():
    Pregnancies = st.sidebar.text_input('Pregnancies')
    #CLMINSUR = st.sidebar.selectbox('Insurance',('1','0'))
    BloodPressure = st.sidebar.text_input("Blood Pressure")
    BMI = st.sidebar.text_input("BMI")
    Age = st.sidebar.text_input("Age")
    Glucose = st.sidebar.text_input("Glucose")
    Insulin = st.sidebar.text_input("Insulin")
    SkinThickness = st.sidebar.text_input("SkinThickness")
    DiabetesPedigreeFunction = st.sidebar.text_input("DiabetesPedigreeFunction")
    
    data = {'Pregnancies':Pregnancies,
            'BloodPressure':BloodPressure,
            'BMI':BMI,
            'Age':Age,
            'Glucose':Glucose,
            'Insulin':Insulin,
            'SkinThickness':SkinThickness,
            'DiabetesPedigreeFunction':DiabetesPedigreeFunction,
           }
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')

st.write(df)

diabetes = pd.read_csv("diabetes.csv")
#claimants.drop(["CASENUM"],inplace=True,axis = 1)
#claimants = claimants.dropna()

X = diabetes.iloc[:,[0,2,5,7,1,4,3,6]]
Y = diabetes.iloc[:,8]
clf = LogisticRegression()
clf.fit(X,Y)

if not df.empty:
    prediction = clf.predict(df)
    prediction_proba = clf.predict_proba(df)

    st.subheader('Predicted Result')
    st.write('Yes' if prediction_proba[0][1] > 0.5 else 'No')

    st.subheader('Prediction Probability')
    st.write(prediction_proba)