import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd

# Function to load the selected model
def load_model_file(model_path, weights_path):
    try:
        loaded_model = load_model(model_path)
        loaded_model.load_weights(weights_path)
        st.success("Model berhasil dimuat.")
        return loaded_model
    except Exception as e:
        st.error(f"Error saat membaca model: {e}")
        return None

# Function to make diabetes prediction
def predict_diabetes(model, input_data):
    if model:
        prediction = model.predict(input_data)
        return prediction[0]
    return None

# Function to display the prediction result
def display_diagnosis(prediction):
    if prediction is not None:
        # Check if it's a binary classification or multiclass classification
        if len(prediction) == 1:  # Binary classification
            return f'Orang tersebut terkena diabetes dengan nilai prediksi {prediction[0]:.2f}' if prediction[0] >= 0.7 else f'Orang tersebut bebas diabetes dengan nilai prediksi {prediction[0]:.2f}'
        else:  # Multiclass classification
            predicted_class = np.argmax(prediction)  # Get the index of the predicted class
            return f'Orang tersebut terkena diabetes (Kelas {predicted_class}) dengan nilai prediksi {prediction[predicted_class]:.2f}'
    return ''

# Function to save prediction history
def save_prediction_history(history, entry):
    return history.append(entry, ignore_index=True)

# Function to fetch prediction history
def fetch_prediction_history(session_state):
    if 'prediction_history' not in session_state:
        return pd.DataFrame(columns=['Input Data', 'Prediction', 'Diagnosis'])
    return session_state.prediction_history

# Streamlit app
st.title('Prediksi Diabetes Menggunakan Neural Network')

# Load models
model_options = ['Model_ann', 'Model_cnn']
selected_model = st.radio('Pilih Model:', model_options)

model_folder = 'E:\\syntax code\\python\\jupytr\\neural network\\diabetes\\model\\'
if selected_model == 'Model_ann':
    weight_path = model_folder + 'weights_ann_new.h5'
    model_path = model_folder + 'model_ann_new.h5'
elif selected_model == 'Model_cnn':
    weight_path = model_folder + 'diabetes_weight_DNN.h5'
    model_path = model_folder + 'diabetes_DNN.h5'

diabetes_model = load_model_file(model_path, weight_path)

# Input form
input_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
input_data = [st.text_input(f'{column}:') for column in input_columns]

# Handling empty string inputs
input_data = [float(val) if val != '' else 0.0 for val in input_data]

# Convert to NumPy array
input_data = np.array(input_data).reshape(1, -1)

# Prediction and result
diagnosis = ''
if st.button('Diabetes Test Result'):
    prediction = predict_diabetes(diabetes_model, input_data)
    if prediction is not None:
        diagnosis = display_diagnosis(prediction)

        # Display diagnosis
        st.success(diagnosis)
        
        # Display prediction values
        st.subheader('Nilai Prediksi')
        st.write(np.around(prediction, decimals=2))

        # Save to prediction history
        entry = {'Input Data': input_data.flatten(), 'Prediction': prediction.tolist(), 'Diagnosis': diagnosis}
        st.session_state.prediction_history = save_prediction_history(fetch_prediction_history(st.session_state), entry)
    else:
        st.error("Error dalam melakukan prediksi.")

# Display prediction history
history_df = fetch_prediction_history(st.session_state)
if not history_df.empty:
    st.subheader('Riwayat Prediksi')
    st.table(history_df)

    # Option to delete selected rows
    rows_to_delete = st.multiselect('Pilih baris yang ingin dihapus', history_df.index)
    if st.button('Hapus Baris Terpilih'):
        if rows_to_delete:
            history_df = history_df.drop(rows_to_delete)
            st.session_state.prediction_history = history_df  # Update session state
            st.success(f"Baris {', '.join(map(str, rows_to_delete))} berhasil dihapus dari riwayat prediksi.")
        else:
            st.warning("Silakan pilih setidaknya satu baris untuk dihapus.")
else:
    st.warning("Belum ada riwayat prediksi.")
