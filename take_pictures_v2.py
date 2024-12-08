import pyrealsense2 as rs
import cv2
import os
import numpy as np
import time

# Configurações
output_dir = "dataset_images"  # Pasta destino das imagens
image_prefix = "rgb_"          # Prefixo de nomenclatura dos arquivos

# Garantir que o diretório de saída existe
os.makedirs(output_dir, exist_ok=True)

# Função para obter o próximo índice de imagem
def get_next_image_index(output_dir, prefix):
    existing_files = [f for f in os.listdir(output_dir) if f.startswith(prefix) and f.endswith(".jpg")]
    indices = [int(f[len(prefix):-4]) for f in existing_files if f[len(prefix):-4].isdigit()]
    return max(indices, default=0) + 1

# Solicitar ao usuário o tempo de gravação
while True:
    try:
        duration_seconds = int(input("Quantos segundos deseja capturar frames? "))
        if duration_seconds > 0:
            break
        else:
            print("Por favor, insira um número maior que zero.")
    except ValueError:
        print("Entrada inválida. Por favor, insira um número inteiro.")

# Configuração da câmera RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# Inicializar contadores
start_time = time.time()
frame_count = 0
next_index = get_next_image_index(output_dir, image_prefix)

try:
    print("Capturando frames... Pressione 'Q' para sair antecipadamente.")
    while time.time() - start_time < duration_seconds:
        # Captura de frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # Converter frames para arrays numpy
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Salvar a imagem RGB
        rgb_name = f"{image_prefix}{next_index + frame_count}.jpg"
        rgb_path = os.path.join(output_dir, rgb_name)
        cv2.imwrite(rgb_path, color_image)

        # Salvar a imagem de profundidade
        depth_name = f"depth_{image_prefix}{next_index + frame_count}.png"
        depth_path = os.path.join(output_dir, depth_name)
        cv2.imwrite(depth_path, depth_image)

        # Salvar as coordenadas 3D em um arquivo .txt
        txt_name = f"points_3d_{image_prefix}{next_index + frame_count}.txt"
        txt_path = os.path.join(output_dir, txt_name)
        np.savetxt(txt_path, depth_image, fmt='%d')

        frame_count += 1

        # Exibir a imagem RGB
        cv2.imshow("Captura de Imagem RGB", color_image)

        # Sair ao pressionar 'Q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time
    print(f"Captura concluída. Taxa média de FPS: {avg_fps:.2f}")

finally:
    # Encerrar pipeline e janelas
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Frames processados e salvos com sucesso.")
