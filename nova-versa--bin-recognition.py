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

    # Configurar os dois modelos YOLO
    specific_model = YOLO("/home/iafa/runs/detect/train9/weights/best.pt")  # Modelo específico
    generic_model = YOLO('yolov8n.pt')  # Modelo genérico pré-treinado para validação

    # Matriz de transformação da câmera para a base do robô
    T_camera_to_base = np.array([
        [0, -1,  0, 0.8853],
        [1,  0,  0, 0.0465],
        [0,  0,  1, -0.2823],
        [0,  0,  0, 1]
    ])

    print("Iniciando detecção em tempo real. Pressione 'Q' para sair.")
    last_output = None  # Armazena os últimos parâmetros exibidos no terminal

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
        specific_results = specific_model.predict(frame, verbose=False)
        generic_results = generic_model.predict(frame, verbose=False)

        # Processar as detecções do modelo específico
        for result in specific_results:
            for box in result.boxes:
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = box.conf[0]

                class_name = specific_model.names[class_id]

                # Verificar se é OverheadBin
                if class_name == "OverheadBin":
                    # Validar com IOU usando o modelo genérico
                    is_valid = True
                    for gen_result in generic_results:
                        for gen_box in gen_result.boxes:
                            x_min_gen, y_min_gen, x_max_gen, y_max_gen = map(int, gen_box.xyxy[0])
                            gen_class_id = int(gen_box.cls[0])
                            gen_confidence = gen_box.conf[0]
                            gen_class_name = generic_model.names[gen_class_id]

                            # Calcular IOU e verificar confiança
                            iou_value = iou((x_min, y_min, x_max, y_max), (x_min_gen, y_min_gen, x_max_gen, y_max_gen))
                            if iou_value > 0.85 and gen_confidence > confidence:
                                print(f"OverheadBin descartado devido ao alto IOU com {gen_class_name} ({iou_value:.2f}) e maior confiança ({gen_confidence:.2f} > {confidence:.2f}).")
                                is_valid = False
                                break

                        if not is_valid:
                            break

                    if not is_valid:
                        continue

                    # Cálculo das coordenadas reais
                    x_center = int((x_min + x_max) / 2)
                    y_center = int((y_min + y_max) / 2)
                    distance = depth_frame.get_distance(x_center, y_center)  # Distância em metros

                    X_real = (x_center - cx) * distance / fx
                    Y_real = (y_center - cy) * distance / fy
                    Z_real = distance  # Usando o valor de distância como Z

                    # Verificação de repetição com tolerância de 11%
                    current_output = (round(X_real, 2), round(Y_real, 2), round(Z_real, 2))
                    if last_output is not None:
                        same_as_last = (
                            np.isclose(current_output[0], last_output[0], rtol=0.11) and
                            np.isclose(current_output[1], last_output[1], rtol=0.11) and
                            np.isclose(current_output[2], last_output[2], rtol=0.11)
                        )
                    else:
                        same_as_last = False

                    if not same_as_last:
                        # Exibir coordenadas em relação à câmera
                        print(f"As coordenadas do OverheadBin relativas ao sistema de ref. da câmera são: X={X_real:.4f}, Y={Y_real:.4f}, Z={Z_real:.4f}")

                        # Transformar para o sistema de coordenadas da base
                        P_camera = np.array([X_real, Y_real, Z_real, 1]).reshape(4, 1)
                        P_base = np.dot(T_camera_to_base, P_camera)
                        X_base, Y_base, Z_base = P_base[:3].flatten()

                        # Exibir coordenadas em relação à base
                        print(f"As coordenadas do OverheadBin relativas ao sistema de ref. da base do robô são: X={X_base:.2f}, Y={Y_base:.2f}, Z={Z_base:.2f}")
                        
                        last_output = current_output

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
