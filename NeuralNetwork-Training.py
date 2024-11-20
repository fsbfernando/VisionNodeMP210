# Inicialização para uso de Roboflow e Yolo
from ultralytics import YOLO
import os
from roboflow import Roboflow

# Função para limpeza do console
def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

clear_console()

# Obtendo meu dataset via Roboflow em formato compatível com yolov8
rf = Roboflow(api_key="72NztabwEW1Ulqazqbhe")
project = rf.workspace("fernando-braga").project("officebox")
version = project.version(3)
dataset = version.download("yolov8")

# Caminho para os dados
data_path = os.path.join(dataset.location, "data.yaml")

# Sinalização via prompt - início do treinamento
print("Iniciando o treinamento do modelo YOLO...")

# Inicializando o modelo YOLOv8 - Versão Nano 
model = YOLO("yolov8n.pt")  

# Definição de local de armazenamento, número de épocas, tamanho do batch, e tamanho das imagens utilizadas
model.train(
    data=data_path,  
    epochs=22,      
    batch=4,        
    imgsz=(640, 480) 
)

# Sinalização via prompt - Fim do treinamento
print("Treinamento concluído com sucesso!")
