# Importación de librerías necesarias para el control del dron, la identificacion de gestos y apriltags
from djitellopy import Tello
import cv2, time, os, sys, signal, platform
import numpy as np
from datetime import datetime
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pupil_apriltags import Detector

#            CONFIGURACIÓN DE GESTOS 
# Define los pares de puntos (landmarks) para dibujar el esqueleto de la mano en pantalla
CONEXIONES_MANO = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),         # Índice
    (5, 9), (9, 10), (10, 11), (11, 12),    # Medio
    (9, 13), (13, 14), (14, 15), (15, 16),  # Anular
    (13, 17), (17, 18), (18, 19), (19, 20), # Meñique
    (0, 17)                                 # Palma
]

# Clase encargada de manejar el modelo de gesture_recognizeer.task para detectar manos y gestos
class GestureDetector:
    def __init__(self, model_path='gesture_recognizer.task'):
        # Configuración base del modelo de Mediapipe
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            # Se configura en modo IMAGE para procesar cuadro por cuadro de forma independiente
            running_mode=vision.RunningMode.IMAGE,
            min_hand_detection_confidence=0.5, # Confianza mínima para detectar una mano
            min_hand_presence_confidence=0.5,  # Confianza mínima para confirmar que la mano sigue ahí
            min_tracking_confidence=0.5,       # Confianza mínima para el seguimiento
            num_hands=1                        # Número máximo de manos a analizar simultáneamente
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)
        # Variable para almacenar el nombre en crudo del último gesto detectado (ej. Open_Palm)
        self.ultimo_gesto = "Ninguno" 

    # Función para dibujar los puntos (nodos) y líneas (huesos) de la mano sobre el frame de video
    def draw_manual_landmarks(self, frame, landmarks):
        h, w, _ = frame.shape
        # Convierte las coordenadas normalizadas del modelo a píxeles exactos de la pantalla
        puntos_px = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        # Dibuja las líneas que conectan los dedos
        for conexion in CONEXIONES_MANO:
            p1, p2 = puntos_px[conexion[0]], puntos_px[conexion[1]]
            cv2.line(frame, p1, p2, (255, 255, 255), 2)
        # Dibuja los puntos en cada articulación
        for pt in puntos_px:
            cv2.circle(frame, pt, 5, (255, 255, 0), -1)

    # Función matemática para interpretar direcciones basándose en la posición del dedo índice
    def get_direction(self, landmarks):
        # Compara la punta del índice (punto 8) con la base del índice (punto 5)
        idx_tip, idx_base = landmarks[8], landmarks[5]
        dx, dy = idx_tip.x - idx_base.x, idx_tip.y - idx_base.y
        umbral = 0.06 # Distancia mínima requerida para considerar que hay un movimiento claro e intencional
        
        # Determina si la inclinación del dedo es principalmente horizontal o vertical
        if abs(dx) > abs(dy):
            if dx > umbral: return "DERECHA"
            elif dx < -umbral: return "IZQUIERDA"
        else:
            if dy > umbral: return "ABAJO"
            elif dy < -umbral: return "ARRIBA"
        return None

    # Función principal de la clase que recibe la imagen y devuelve la imagen procesada junto con el comando
    def process_frame(self, frame):
        # Convierte la imagen al formato compatible que requiere Mediapipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Ejecuta la red neuronal para reconocer el gesto
        result = self.recognizer.recognize(mp_image)
        label = "Esperando..."
        self.ultimo_gesto = "Ninguno" 

        # Si el modelo detecta puntos clave de una mano en la imagen...
        if result.hand_landmarks:
            self.draw_manual_landmarks(frame, result.hand_landmarks[0])
            gesture_name = result.gestures[0][0].category_name
            self.ultimo_gesto = gesture_name 
            
            # Mapeo directo del gesto detectado a una acción o comando de vuelo
            if gesture_name == "Open_Palm": label = "DESPEGAR"
            elif gesture_name == "Closed_Fist": label = "ATERRIZAR"
            elif gesture_name == "Thumb_Up": label = "FLIP ATRAS"
            elif gesture_name == "Thumb_Down": label = "ABAJO"      # Este gesto esta pero no ejecuta nada
            elif gesture_name == "Victory": label = "MODO TRACKING" # Gatillo que activa el seguimiento de AprilTags
            elif gesture_name == "ILoveYou": label = "Rutina"       # Gatillo que activa una coreografía de movimientos
            else:
                # Si no es un gesto estático configurado, intenta leer una dirección cardinal con el índice
                dir_label = self.get_direction(result.hand_landmarks[0])
                label = dir_label if dir_label else "Esperando..."

        return frame, label

# =======================
# TECLADO (SO Respaldo)
# =======================
# Lógica para detectar qué sistema operativo se usa y configurar el control manual por teclado adecuadamente
OS = platform.system()
USE_PYNPUT = (OS == "Darwin") # En MacOs se requiere la librería pynput  
if USE_PYNPUT:
    from pynput import keyboard # type: ignore
    keys = set() # Set para almacenar las teclas presionadas en tiempo real
    def on_press(key):
        try: keys.add(key.char)
        except: 
            if key == keyboard.Key.esc: keys.add('esc')
    def on_release(key):
        try: keys.discard(key.char)
        except: 
            if key == keyboard.Key.esc: keys.discard('esc')
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

# =======================
# CONFIGURACIÓN DRON Y CÁMARA
# =======================
# Inicializa el objeto principal de la librería del dron y establece conexión WiFi
tello = Tello()
tello.connect()
print("Batería Tello:", tello.get_battery(), "%")

# Apaga y vuelve a encender el stream de video para asegurar un inicio limpio
tello.streamoff()
tello.streamon()
frame_read = tello.get_frame_read() # Objeto que extrae el video en tiempo real del dron
time.sleep(2) # Tiempo de espera para que la cámara del dron estabilice la señal

# Función de seguridad para detener motores forzosamente y aterrizar
def safe_land():
    print("Aterrizaje de emergencia...")
    for _ in range(3):
        tello.send_rc_control(0,0,0,0) # Envía velocidades nulas a todos los ejes para frenarlo
        time.sleep(0.05)
    tello.land()
    sys.exit(0)

# Intercepta el atajo de teclado de consola "Ctrl+C" para aterrizar de manera segura en vez de colapsar
signal.signal(signal.SIGINT, lambda sig, frame: safe_land())

# =======================
# VARIABLES PRINCIPALES
# =======================
# Instanciamiento de los modelos de visión artificial
detector_gestos = GestureDetector()
detector_tags = Detector(families="tag36h11") # Se configura el modelo de marcadores fiduciarios (AprilTags)

# Watchdog / Temporizador límite para mantenerse dentro del modo de seguimiento autónomo
tiempo_limite_tags = 0 

# Velocidad de desplazamiento por defecto para cuando se controla manualmente con gestos
speed_manual = 30 
last_rc_time = 0
rc_interval = 0.05 # Límite de envío de paquetes (para no inundar de comandos al dron)

# --- PARÁMETROS DE SEGUIMIENTO (TRACKING PID) ---
# Variables de control Proporcional para mantener al dron estabilizado frente a un objetivo visual
TARGET_SIZE = 120  # Tamaño ideal del tag en píxeles (dicta a qué distancia de profundidad se debe mantener el dron)
KP_YAW = 0.3       # Ganancia proporcional para el eje de rotación (Yaw)
KP_UD = 0.4        # Ganancia proporcional para la altitud (Up/Down)
KP_FB = 0.4        # Ganancia proporcional para el acercamiento/alejamiento (Forward/Backward)

# Variables de estado del vuelo y de control del ciclo de frames
volando=False
tiempo_inicio = time.time()
counter=0
while True:
    frame = frame_read.frame
    # frame+=10
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Corrección de canal de colores para la IA
    if frame is None: continue

    # Extrae el tamaño de la imagen para determinar el punto central geométrico de la cámara
    alto_cam, ancho_cam, _ = frame.shape
    centro_cam_x = ancho_cam // 2
    centro_cam_y = alto_cam // 2
    
    tiempo_actual = time.time()
    
    # Evalúa si la cuenta regresiva del modo autónomo sigue activa
    modo_tags_activo = tiempo_actual < tiempo_limite_tags

    # Velocidades de canal de control de vuelo inicializadas en cero para cada frame (Left/Right, Forward/Backward, Up/Down, Yaw)
    lr, fb, ud, yaw = 0, 0, 0, 0

    if modo_tags_activo:
        # ==========================================
        # MODO 1: SEGUIMIENTO AUTÓNOMO DE APRILTAGS
        # ==========================================
        # El detector de tags de pupillabs necesita trabajar sobre una imagen en escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resultado_tags = detector_tags.detect(gray)
        
        apriltags_detectado = False

        # Si el modelo encontró al menos un AprilTag...
        if len(resultado_tags) > 0:
            tag = resultado_tags[0] # Siempre selecciona el primero encontrado para seguirlo
            apriltags_detectado = True
            
            # Al estar viendo el tag, se resetea y extiende el tiempo para no salir del modo autónomo
            tiempo_limite_tags = tiempo_actual + 7.0 # Si deja de ver el tag y la cuenta regresiva llega a cero se activan los gestos

            # 1. Calcular el punto central detectado del Tag y estimar su tamaño aparente promediando ancho y alto
            cx, cy = int(tag.center[0]), int(tag.center[1])
            width = np.linalg.norm(tag.corners[0] - tag.corners[1])
            height = np.linalg.norm(tag.corners[1] - tag.corners[2])
            size_px = (width + height) / 2

            # 2. Calcular los Errores: la distancia en píxeles entre el centro de la cámara y el centro/tamaño del Tag
            error_x = cx - centro_cam_x
            error_y = cy - centro_cam_y
            error_size = TARGET_SIZE - size_px

            # 3. Calcular los comandos RC Proporcionales (Multiplicando error * Ganancia y bloqueando la velocidad a máximo 50)
            yaw = int(np.clip(error_x * KP_YAW, -50, 50))
            ud = int(np.clip(-error_y * KP_UD, -50, 50)) 
            fb = int(np.clip(error_size * KP_FB, -50, 50))

            # Zonas muertas: Si el dron está muy cerca del centro deseado, asigna velocidad cero para evitar temblores excesivos (jitter)
            if abs(error_x) < 20: yaw = 0
            if abs(error_y) < 20: ud = 0
            if abs(error_size) < 15: fb = 0

            # Dibujar retículas e indicadores visuales de acople en la pantalla
            cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 25, 2)
            cv2.rectangle(frame, (centro_cam_x - 120, alto_cam - 70), (centro_cam_x + 120, alto_cam - 20), (0, 120, 0), -1)
            cv2.putText(frame, "SIGUIENDO", (centro_cam_x - 100, alto_cam - 35), 
                        cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 2)

        # Dibujar recuadro que muestra los segundos restantes antes de cancelar el modo de seguimiento
        tiempo_restante = max(0, tiempo_limite_tags - tiempo_actual)
        cv2.rectangle(frame, (0, 0), (500, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"BUSCANDO TAG: {tiempo_restante:.1f}s", (20, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if not apriltags_detectado:
            # Si se perdió de vista el Tag en este frame, frena por completo el dron (estado de Hover o reposo activo)
            lr, fb, ud, yaw = 0, 0, 0, 0
            cv2.putText(frame, "SIN CONTACTO", (ancho_cam - 200, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    else:
        # ==========================================
        # MODO 2: CONTROL MANUAL POR GESTOS
        # ==========================================
        # Delega a la IA la detección de la mano y extrae la etiqueta interpretada
        frame, etiqueta = detector_gestos.process_frame(frame)
        
        # Dibuja la instrucción manual obtenida en la parte superior de la imagen
        cv2.rectangle(frame, (0, 0), (450, 60), (0, 0, 0), -1)
        cv2.putText(frame, etiqueta, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Lógica para procesar y accionar las directivas indicadas por la etiqueta manual
        if etiqueta == "MODO TRACKING":                 # Se activa al usar el gesto de 'VICTORY'
            tiempo_limite_tags = tiempo_actual + 7.0    # Si dectecta el tag siempre se mantiene en 7 y si no lo detecta despues de 7 segundos se activan los gestos
        
        elif etiqueta == "DESPEGAR" and volando==False: # Se activa al usar el gesto de 'Open_Palm'
            tello.takeoff()
            volando=True
            time.sleep(2) 
       
        elif etiqueta == "ATERRIZAR":                   # Se activa al usar el gesto de 'Closed_Fist'
            safe_land()
            time.sleep(1)
        
        elif etiqueta == "FLIP ATRAS":                  # Se activa al usar el gesto de 'Thumb_Up'
            try: tello.flip_back()
            except: pass
            time.sleep(2) 
        
        # Inyección continua de la variable base de velocidad de dirección (Ejecuta el gesto en el tiempo basado en la direccion del indice)
        elif etiqueta == "ARRIBA": ud = speed_manual
        elif etiqueta == "ABAJO": ud = -speed_manual
        elif etiqueta == "DERECHA": lr = speed_manual
        elif etiqueta == "IZQUIERDA": lr = -speed_manual
        
        if etiqueta == "Rutina":# Cuando detecta gesto 'ILoveYou', ejecuta el la rutina
            # Rutina de 7 movimientos(el dron hará esta coreografía ignorando otros inputs temporalmente)
            tello.move_back(50)
            tello.move_forward(50)
            tello.rotate_clockwise(360)
            tello.move_left(50)
            tello.move_right(50)
            tello.rotate_counter_clockwise(360)
            tello.move_back(60)

    # Imprime en tiempo real los datos proporcionados por los sensores internos del dron en la zona superior derecha
    cv2.putText(frame, f"Bat: {tello.get_battery()}%", (ancho_cam - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Time: {int(time.time() - tiempo_inicio)}s", (ancho_cam - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Alt: {tello.get_height()}cm", (ancho_cam - 150, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # =======================
    # TECLADO (Sobrescribe gestos)
    # =======================
    # Permite al desarrollador intervenir manualmente forzando velocidades con comandos en consola usando estilo WASD
    if USE_PYNPUT:
        cv2.pollKey()
        pressed = keys.copy()
    else:   
        key = cv2.waitKey(1) & 0xFF
        pressed = set()
        if key != 255: pressed.add(chr(key))

    # Actualiza las variables de desplazamiento analizando qué tecla fue accionada
    if 'w' in pressed: fb = speed_manual
    if 's' in pressed: fb = -speed_manual
    if 'a' in pressed: lr = -speed_manual
    if 'd' in pressed: lr = speed_manual
    if 'r' in pressed: ud = speed_manual
    if 'f' in pressed: ud = -speed_manual
    if 'q' in pressed: yaw = -speed_manual
    if 'e' in pressed: yaw = speed_manual

    # Interpreta los valores finales calculados para las velocidades e infiere qué movimiento de traslación se está realizando
    accion_actual = "Hover (Manteniendo Posicion)"
    if ud > 0: accion_actual = "Subiendo"
    elif ud < 0: accion_actual = "Bajando"
    elif lr > 0: accion_actual = "Moviendo Derecha"
    elif lr < 0: accion_actual = "Moviendo Izquierda"
    elif fb > 0: accion_actual = "Avanzando"
    elif fb < 0: accion_actual = "Retrocediendo"
    elif yaw > 0: accion_actual = "Rotando Derecha"
    elif yaw < 0: accion_actual = "Rotando Izquierda"
    
    # Extrae exactamente qué leyó el algoritmo de gestos de Mediapipe (si no está en seguimiento autónomo)
    gesto_mostrar = detector_gestos.ultimo_gesto if not modo_tags_activo else "Modo Autonomo"
    
    # Valida y reemplaza la cadena de movimiento si actualmente se gatilló una maniobra estática superior como aterrizar
    if not modo_tags_activo:
        try:
            if etiqueta in ["DESPEGAR", "ATERRIZAR", "FLIP ATRAS", "FLIP ADELANTE", "Rutina"]:
                accion_actual = etiqueta
        except NameError:
            pass

    # UI: Dibujar recuadro de lectura cruda y estado actual de acción en la esquina inferior izquierda
    cv2.rectangle(frame, (5, alto_cam - 95), (420, alto_cam - 10), (0, 0, 0), -1)
    cv2.putText(frame, f"Gesto Raw: {gesto_mostrar}", (15, alto_cam - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 255), 2)
    cv2.putText(frame, f"Accion: {accion_actual}", (15, alto_cam - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

    # =======================
    # ENVIAR COMANDOS AL DRON
    # =======================
    # Usa un limitador temporal (rc_interval) para enviar los valores finales de todas las velocidades al vuelo
    if time.time() - last_rc_time > rc_interval:
        tello.send_rc_control(lr, fb, ud, yaw)
        last_rc_time = time.time()

    # Rutina para escribir un historial fotográfico del vuelo guardando frames individuales en el sistema operativo
    os.makedirs('procesefotos', exist_ok=True)
    cv2.imwrite(f'procesefotos/frame_{counter}.png', frame)
    counter += 1
    
    # Renderiza y actualiza la ventana visual principal del programa
    cv2.imshow("Tello - Inteligencia Dual", frame)

    # 'l' es el boton de emergencia al apretarlo detiene el programa y llama al protocolo de aterrizaje
    if 'l' in pressed or 'esc' in pressed:
        safe_land()
        break

# Cierre seguro del programa
tello.end()
cv2.destroyAllWindows()