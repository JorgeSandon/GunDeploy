import streamlit as st
import cv2
import torch
import numpy as np

st.title('Gun Video Detector')

# Cargar el modelo
model = torch.hub.load("ultralytics/yolov5", "custom", path="C:/Users/LUIS/Desktop/Video/model/best.pt")

# Función para realizar detecciones en el video
def detect_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Error al abrir el video.")
        return

    # Obtener la información del video (ancho, alto y tasa de fotogramas)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    # Definir el codec y crear el objeto VideoWriter
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frame_rate, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Realizar la detección en cada fotograma
        results = model(frame)

        # Obtener los resultados de detección
        pred = results.pandas().xyxy[0]

        # Iterar sobre las detecciones y dibujar cuadros y etiquetas
        for _, det in pred.iterrows():
            label = f'{det["name"]}: {det["confidence"]:.2f}'  # Ajustamos el nombre de la columna
            cv2.rectangle(frame, (int(det["xmin"]), int(det["ymin"])), (int(det["xmax"]), int(det["ymax"])), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(det["xmin"]), int(det["ymin"]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Escribir el fotograma modificado en el video de salida
        out.write(frame)

    # Liberar los recursos
    cap.release()
    out.release()

# Interfaz para subir un archivo de video
uploaded_file = st.file_uploader("Selecciona un archivo de video", type=["mp4"])

if uploaded_file is not None:
    # Guardar el archivo de video en una ubicación temporal
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("Realizando detecciones en el video...")

    # Realizar detecciones en el video cargado
    output_video_path = "output_video.mp4"
    detect_video("temp_video.mp4", output_video_path)

    # Mostrar el video con las detecciones
    st.video(output_video_path)
