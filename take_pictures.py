import pyrealsense2 as rs 
import cv2
import os
import numpy as np
import time
import winsound  # Para emitir o som no Windows

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

# Solicitar ao usuário o número de imagens a capturar
while True:
    try:
        num_images_to_capture = int(input("Quantas imagens deseja capturar? "))
        if num_images_to_capture > 0:
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

try:
    print("Iniciando captura de imagens. Pressione 'Q' para sair.")
    captured_count = 0
    next_index = get_next_image_index(output_dir, image_prefix)

    while captured_count < num_images_to_capture:
        # Captura de frame da câmera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # Converter para arrays numpy
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Exibir a imagem RGB
        cv2.imshow("Captura de Imagem RGB", color_image)

        # Salvar a imagem RGB
        rgb_name = f"{image_prefix}{next_index + captured_count}.jpg"
        rgb_path = os.path.join(output_dir, rgb_name)
        cv2.imwrite(rgb_path, color_image)

        # Salvar a imagem de profundidade
        depth_name = f"depth_{image_prefix}{next_index + captured_count}.png"
        depth_path = os.path.join(output_dir, depth_name)
        cv2.imwrite(depth_path, depth_image)

        # Salvar informações de profundidade em um arquivo .txt
        txt_name = f"points_3d_{image_prefix}{next_index + captured_count}.txt"
        txt_path = os.path.join(output_dir, txt_name)
        np.savetxt(txt_path, depth_image, fmt='%d')
        
        print(f"Imagem {captured_count + 1} capturada e salva.")
        winsound.Beep(1000, 500)  # Emitir som (frequência 1000 Hz por 500 ms)
        
        captured_count += 1

        # Pausa de 4 segundos
        time.sleep(4)

        # Sair ao pressionar 'Q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Encerrar pipeline e janelas
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Captura concluída.")
