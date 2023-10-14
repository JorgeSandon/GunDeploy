# Importación de las librerías
import torch
import cv2
import numpy as np

# Lectura del modelo
model = torch.hub.load("ultralytics/yolov5", "custom",
                       path="C:/Users/LUIS/Desktop/Video/model/best.pt")

# Ruta del video a detectar
video_path = "C:/Users/LUIS/Desktop/Computer vision/Prueba2.mp4"
cap = cv2.VideoCapture(video_path)

# Verificar si la apertura del video fue exitosa
if not cap.isOpened():
    print("Error al abrir el video.")
    exit()

# Obtener la información del video (ancho, alto y tasa de fotogramas)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

# Definir el codec y crear el objeto VideoWriter
out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frame_rate, (frame_width, frame_height))

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

    # Mostrar la imagen con las detecciones
    cv2.imshow('Video con detecciones', frame)

    # Salir del bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
out.release()
cv2.destroyAllWindows()
