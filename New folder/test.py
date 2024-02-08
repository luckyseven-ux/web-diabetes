import streamlit as st
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load the model
model = load_model("E:\\syntax code\\python\\jupytr\\neural network\\diabetes\\model\\model_ann_new.h5")
model.load_weights("E:\\syntax code\\python\\jupytr\\neural network\\diabetes\\model\\weights_ann_new.h5")

# Streamlit app
st.title('Prediksi Diabetes Menggunakan Neural Network')

# Input form
input_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
input_data = [st.text_input(f'{column}:') for column in input_columns]

# Handling empty string inputs and normalization
input_data = [float(val) if val != '' else 0.0 for val in input_data]
scaler = MinMaxScaler()  # Gunakan scaler yang sama yang digunakan selama pelatihan
input_data = scaler.fit_transform(np.array(input_data).reshape(1, -1))

# Prediction and result
if st.button('Prediksi Diabetes'):
    prediction = model.predict(input_data)

    # Set threshold for binary classification (adjust as needed)
    threshold = 0.5
    prediction_result = 1 if prediction[0][0] > threshold else 0

    st.subheader('Hasil Prediksi')
    st.write(f"Prediksi Diabetes: {prediction_result}")
