import streamlit as st
import numpy as np
from tensorflow.keras.models import model_from_json
import json

# Fungsi untuk memuat model yang dipilih
def load_model_file(model_path, weight_path):
    try:
        if model_path.endswith('.json') and weight_path.endswith('.h5'):
            # Baca arsitektur model dari JSON
            with open(model_path, 'r') as json_file:
                model_json = json_file.read()
            
            # Buat model dari JSON
            loaded_model = model_from_json(model_json)
            
            # Muat bobot model
            loaded_model.load_weights(weight_path)
            
            st.success("Model berhasil dimuat.")
            return loaded_model
        else:
            st.error("Format model tidak didukung.")
            return None
    except Exception as e:
        st.error(f"Error saat membaca model: {e}")
        return None

# Fungsi untuk membuat prediksi diabetes
def predict_diabetes(model, input_data):
    if model:
        prediction = model.predict(input_data)
        return prediction
    return None

# Fungsi untuk menampilkan hasil prediksi
def display_diagnosis(prediction):
    if prediction is not None:
        # Menggunakan any() untuk mengecek apakah ada setidaknya satu elemen yang sama dengan 1
        if np.any(prediction[0] == 1):
            return 'Orang tersebut terkena diabetes'
        else:
            return 'Orang tersebut tidak terkena diabetes'
    return ''

# Memuat model
model_options = ['Model_ann', 'Model_cnn']
selected_model = st.radio('Pilih Model:', model_options)

if selected_model == 'Model_ann':
    weight_path = 'E:\\syntax code\\python\\jupytr\\neural network\\diabetes\\model\\model_weights.h5'
    model_path = 'E:\\syntax code\\python\\jupytr\\neural network\\diabetes\\model\\model.json'
elif selected_model == 'Model_cnn':
    weight_path = 'E:\\syntax code\\python\\jupytr\\neural network\\diabetes\\model\\diabetes_weight_ANN.h5'
    model_path = 'E:\\syntax code\\python\\jupytr\\neural network\\diabetes\\model\\diabetes_ANN.json'

diabetes_model = load_model_file(model_path, weight_path)
input_columns = ['Kehamilan', 'Glukosa', 'TekananDarah', 'KetebalanKulit', 'Insulin', 'BMI', 'FungsiPewarisanDiabetes', 'Usia']
input_data = [st.text_input(f'{column}:') for column in input_columns]

# Menangani input string kosong
input_data = [float(val) if val != '' else 0.0 for val in input_data]

# Mengubah menjadi array NumPy
input_data = np.array(input_data).reshape(1, -1)

# Prediksi dan hasil
diagnosis = ''
if st.button('Hasil Tes Diabetes'):
    prediction = predict_diabetes(diabetes_model, input_data)
    diagnosis = display_diagnosis(prediction)

    # Menampilkan nilai prediksi
    st.subheader('Nilai Prediksi')
    st.write(f"Nilai Prediksi: {prediction[0]}")

    # Menampilkan diagnosis
    st.success(diagnosis)
