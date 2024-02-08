import streamlit as st 
import pickle as pk


model_path = 'diabetes_model.sav'
try:
    with open(model_path, 'rb') as model_file:
        diabetes_model = pk.load(model_file)
    print("Model berhasil dimuat.")
except Exception as e:
    print(f"Error saat membaca model: {e}")

st.title('Prediksi Diabetes Menggunakan Neural Network')
    
    
    # getting the input data from the user
col1, col2, col3 = st.columns(3)
    
with col1:
    Pregnancies = st.text_input('Number of Pregnancies')
        
with col2:
    Glucose = st.text_input('Glucose Level')
    
with col3:
    BloodPressure = st.text_input('Blood Pressure value')
    
with col1:
    SkinThickness = st.text_input('Skin Thickness value')
    
with col2:
    Insulin = st.text_input('Insulin Level')
    
with col3:
    BMI = st.text_input('BMI value')
    
with col1:
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
with col2:
    Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
diab_diagnosis = ''
    
    # creating a button for Prediction
    
if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if (diab_prediction[0] == 1):
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'
        
st.success(diab_diagnosis)