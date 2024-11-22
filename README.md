TakePictures: Código Python para tirar fotos com a Intel RealSense D435. Pergunta quantas fotos são desejadas, e realiza essa quantidade de fotos. Para cada foto tirada, é feita uma imagem RGB, uma imagem de profundidade e um arquivo txt com informações 3d dos píxels da imagem.

NeuralNetwork_training: Treina a rede neural a partir do dataset realizado no Roboflow.

x-y-z-real-time: Código que obtém as coordenadas da caixa em relação a câmera, informa se MyBox está sendo detectada e exibe a imagem (com bounding box apenas para MyBox). Esse script utiliza ferramentas incompatíveis com linux.

Real-time-linux: Adaptação de x-y-z-real-time para poder funcionar no linux.

train5: Contém pesos e gráficos gerados pela RN para reconhecer MyBox (Caixa Organizadora para Escritório).

train6: Contém pesos e gráficos gerados pela RN para reconhecer FakeOverheadBin (Armário)

