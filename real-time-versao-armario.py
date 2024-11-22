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
    fake_overhead_bin_weights = "/home/iafa/runs/detect/train5/weights/best.pt"
    model_fake_bin = YOLO(fake_overhead_bin_weights)  # Modelo treinado para FakeOverheadBin
    model_generic = YOLO("yolov8s.pt")  # Modelo genérico para validação cruzada

    print("Iniciando detecção em tempo real. Pressione 'Q' para sair.")
    fake_bin_detected = False  # Controle de detecção

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

        # Realizar detecção com os dois modelos
        results_fake_bin = model_fake_bin.predict(frame, verbose=False)
        results_generic = model_generic.predict(frame, verbose=False)

        current_detection = False  # Controle local para esta iteração

        # Processar as detecções do modelo FakeOverheadBin
        for result in results_fake_bin:
            for box in result.boxes:
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence_fake_bin = box.conf[0]

                # Validar com o modelo genérico para evitar falsas detecções
                blocked = False
                for generic_result in results_generic:
                    for generic_box in generic_result.boxes:
                        if iou(box.xyxy[0], generic_box.xyxy[0]) > 0.5:
                            confidence_generic = generic_box.conf[0]
                            if confidence_generic > 0.9 or (
                                0.8 < confidence_generic <= 0.9 and confidence_generic > confidence_fake_bin
                            ):
                                blocked = True

                # Se não for bloqueado, confirmar a detecção
                if not blocked and model_fake_bin.names[class_id] == "FakeOverheadBin":
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
                        f"FakeOverheadBin: {confidence_fake_bin:.2f}",
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                    current_detection = True
                    if not fake_bin_detected:
                        print(f"FakeOverheadBin detectado a {distance:.4f} metros.")
                        print(f"Coordenadas relativas: X={X_real:.4f}, Y={Y_real:.4f}, Z={distance:.4f}")
                        fake_bin_detected = True

        # Caso o FakeOverheadBin não seja detectado
        if not current_detection and fake_bin_detected:
            print("FakeOverheadBin não está mais visível.")
            fake_bin_detected = False

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
