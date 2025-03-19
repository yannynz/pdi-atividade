import cv2
import numpy as np

# Função para detectar formas geométricas e colisões
def detectar_formas_e_colisoes(frame):
    # Converter para HSV
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Faixas de cores ajustadas
    lower_blue = np.array([90, 50, 50])  # Azul pastel
    upper_blue = np.array([130, 255, 255])
    lower_orange = np.array([10, 100, 100])  # Laranja pastel
    upper_orange = np.array([25, 255, 255])

    # Criar máscaras
    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
    mask_orange = cv2.inRange(img_hsv, lower_orange, upper_orange)

    # Encontrar contornos
    contornos_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos_orange, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Variáveis para armazenar informações das formas
    maior_contorno_azul = None
    maior_area_azul = 0
    maior_contorno_laranja = None
    maior_area_laranja = 0

    # Identificar o maior contorno azul
    for contorno in contornos_blue:
        area = cv2.contourArea(contorno)
        if area > maior_area_azul:
            maior_area_azul = area
            maior_contorno_azul = contorno

    # Identificar o maior contorno laranja
    for contorno in contornos_orange:
        area = cv2.contourArea(contorno)
        if area > maior_area_laranja:
            maior_area_laranja = area
            maior_contorno_laranja = contorno

    # Inicializar variáveis de colisão e ultrapassagem
    colisao_ocorreu = False
    passou_barreira = False

    # Processar contornos e verificar colisões
    if maior_contorno_azul is not None:
        x_blue, y_blue, w_blue, h_blue = cv2.boundingRect(maior_contorno_azul)
        cv2.rectangle(frame, (x_blue, y_blue), (x_blue + w_blue, y_blue + h_blue), (0, 255, 0), 2)  # Retângulo verde

        if maior_contorno_laranja is not None:
            x_orange, y_orange, w_orange, h_orange = cv2.boundingRect(maior_contorno_laranja)
            cv2.rectangle(frame, (x_orange, y_orange), (x_orange + w_orange, y_orange + h_orange), (0, 165, 255), 2)  # Retângulo laranja

            # Verificar colisão entre os bounding boxes
            if (x_blue < x_orange + w_orange and x_blue + w_blue > x_orange and
                y_blue < y_orange + h_orange and y_blue + h_blue > y_orange):
                colisao_ocorreu = True  # Colisão detectada

            # Verificar se o laranja passou a barreira após a colisão
            if colisao_ocorreu and (
                y_orange + h_orange < y_blue or y_orange > y_blue + h_blue or
                x_orange + w_orange < x_blue or x_orange > x_blue + w_blue):
                passou_barreira = True

    return frame, colisao_ocorreu, passou_barreira

# Função principal para processar o vídeo
def processar_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Erro: Não foi possível abrir o vídeo.")
        return

    colisao_ocorreu = False

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Vídeo finalizado ou erro na leitura do frame.")
            break

        # Redimensionar para performance
        altura, largura = frame.shape[:2]
        nova_largura = 640
        nova_altura = int(altura * nova_largura / largura)
        frame_resized = cv2.resize(frame, (nova_largura, nova_altura))

        # Detectar formas e colisões
        frame_processado, colisao, ultrapassagem = detectar_formas_e_colisoes(frame_resized)

        # Exibir mensagens de colisão ou ultrapassagem
        font = cv2.FONT_HERSHEY_SIMPLEX
        if colisao and not ultrapassagem:
            cv2.putText(frame_processado, "COLISAO DETECTADA", (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        elif ultrapassagem:
            cv2.putText(frame_processado, "PASSOU BARREIRA", (50, 100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Mostrar vídeo processado
        cv2.imshow("Video Processado", frame_processado)

        # Verificar se a tecla 'q' foi pressionada para sair
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Caminho do vídeo
video_path = "./q1B.mp4"
processar_video(video_path)

