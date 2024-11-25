import os
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time

# Função IOU para calcular a interseção sobre união
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

try:
    # Iniciar o pipeline
    profile = pipeline.start(config)

    # Obter os parâmetros intrínsecos da câmera
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy

    # Configurar os modelos YOLO
    model_weights = "/home/iafa/runs/detect/train9/weights/best.pt"
    model = YOLO(model_weights)  # Modelo treinado para FakeOverheadBin e OverheadBin

    print("Iniciando detecção em tempo real. Pressione 'Q' para sair.")
    overhead_bin_detected = False  # Controle de detecção

    # Criar uma única janela para exibir a câmera
    cv2.namedWindow("Detecção em Tempo Real", cv2.WINDOW_NORMAL)

    while True:
        # Capturar frames da câmera com timeout de 10 segundos
        frames = pipeline.wait_for_frames(timeout_ms=10000)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # Verificar se os frames são válidos
        if not color_frame or not depth_frame:
            continue

        # Converter frames para arrays numpy
        frame = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Realizar detecção com o modelo treinado
        results = model.predict(frame, verbose=False)

        current_detection = False  # Controle local para esta iteração

        # Processar as detecções do modelo
        for result in results:
            for box in result.boxes:
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = box.conf[0]

                class_name = model.names[class_id]

                # Verificar se é OverheadBin e validar com FakeOverheadBin
                if class_name == "OverheadBin":
                    # Verificar se há alta confiança de FakeOverheadBin
                    fake_bin_confidence = max(
                        box.conf[0]
                        for box in result.boxes
                        if model.names[int(box.cls[0])] == "FakeOverheadBin"
                    ) if any(model.names[int(box.cls[0])] == "FakeOverheadBin" for box in result.boxes) else 0

                    # Descarta se FakeOverheadBin tiver alta confiança
                    if fake_bin_confidence > 0.85:
                        continue

                    x_center = int((x_min + x_max) / 2)
                    y_center = int((y_min + y_max) / 2)
                    distance = depth_frame.get_distance(x_center, y_center)  # Distância em metros

                    # Convertendo coordenadas de pixels para coordenadas reais
                    X_real = (x_center - cx) * distance / fx
                    Y_real = (y_center - cy) * distance / fy

                    # Desenhar bounding box e exibir informações
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"OverheadBin: {confidence:.2f}",
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                    current_detection = True
                    if not overhead_bin_detected:
                        print(f"OverheadBin detectado a {distance:.4f} metros.")
                        print(f"Coordenadas relativas: X={X_real:.4f}, Y={Y_real:.4f}, Z={distance:.4f}")
                        overhead_bin_detected = True

        # Caso o OverheadBin não seja detectado
        if not current_detection and overhead_bin_detected:
            print("OverheadBin não está mais visível.")
            overhead_bin_detected = False

        # Mostrar o frame atualizado apenas uma vez por ciclo
        cv2.imshow("Detecção em Tempo Real", frame)

        # Pressione 'Q' para sair
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except rs.error as e:
    print(f"Erro RealSense: {e}")
except Exception as e:
    print(f"Erro geral: {e}")
finally:
    # Parar o pipeline e fechar janelas
    pipeline.stop()
    cv2.destroyAllWindows()
