import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model

import platform

st.write("Versión de Python:", platform.python_version())

model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Cargar los nombres de las clases desde labels.txt
with open('labels.txt', 'r') as f:
    class_names = [line.strip().split(' ', 1)[1] if ' ' in line else line.strip() for line in f.readlines()]

st.title("Reconocimiento de Imágenes")
image = Image.open('OIG5.jpg')
st.image(image, width=350)
with st.sidebar:
    st.subheader("Usando un modelo entrenado en teachable Machine puedes Usarlo en esta app para identificar")
img_file_buffer = st.camera_input("Toma una Foto")

if img_file_buffer is not None:
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    img = Image.open(img_file_buffer)

    newsize = (224, 224)
    img = img.resize(newsize)
    img_array = np.array(img)

    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)
    print(prediction)

    # Índice de la clase con mayor probabilidad
    idx = np.argmax(prediction[0])
    clase_detectada = class_names[idx]
    probabilidad = prediction[0][idx]

    st.header(f'Detectado: {clase_detectada}, con Probabilidad: {probabilidad}')

    # Verificar si la clase detectada es "samuel"
    if clase_detectada.lower() == "samuel" and probabilidad > 0.5:
        st.success("Esta samuel")
    else:
        st.error("No esta samuel")
