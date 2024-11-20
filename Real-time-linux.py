# Inicialização do script
import os
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time

# Definição de funções
# Função iou
def iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area

# Configuração da câmera RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

# Obtendo os parâmetros intrínsecos da câmera
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
fx = intrinsics.fx
fy = intrinsics.fy
cx = intrinsics.ppx
cy = intrinsics.ppy

# Modelos YOLO
model_mybox = YOLO(r"/Users/fsbfe/runs/detect/train4/weights/best.pt")  # Modelo MyBox Fernando
model_generic = YOLO('yolov8s')                                         # Modelo de validação cruzada

# Variável para controlar a detecção de MyBox
mybox_detected = False

try:
    print("Iniciando detecção em tempo real. Pressione 'Q' para sair.")
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        results_mybox = model_mybox.predict(frame, verbose=False)
        results_generic = model_generic.predict(frame, verbose=False)

        current_detection = False

        for result in results_mybox:
            for box in result.boxes:
                if model_mybox.names[int(box.cls[0])] == 'MyBox':
                    mybox_confidence = box.conf[0]
                    x_min, y_min, x_max, y_max = box.xyxy[0]
                    blocked = False

                    for generic_result in results_generic:
                        for generic_box in generic_result.boxes:
                            if iou(box.xyxy[0], generic_box.xyxy[0]) > 0.5:
                                generic_confidence = generic_box.conf[0]
                                if generic_confidence > 0.9 or (0.8 < generic_confidence <= 0.9 and generic_confidence > mybox_confidence):
                                    blocked = True

                    if not blocked:
                        x_center = int((x_min + x_max) / 2)
                        y_center = int((y_min + y_max) / 2)
                        distance = depth_frame.get_distance(x_center, y_center)  # Distância em metros

                        # Convertendo coordenadas de pixels para metros
                        X_real = (x_center - cx) * distance / fx
                        Y_real = (y_center - cy) * distance / fy

                        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                        cv2.putText(frame, f'MyBox Confirmed', (int(x_min), int(y_min - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        current_detection = True
                        if not mybox_detected:
                            print(f"MyBox está sendo detectada e se encontra a {distance:.4f} metros da câmera.")
                            print(f"Suas coordenadas em relação ao sistema de referência da câmera são: ({X_real:.4f}, {Y_real:.4f}, {distance:.4f})")
                            mybox_detected = True

        if not current_detection and mybox_detected:
            print("MyBox não está sendo detectada")
            mybox_detected = False

        cv2.imshow('Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
