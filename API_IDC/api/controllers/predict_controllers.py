import numpy as np
import tensorflow as tf
from PIL import Image
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

# Localizar la carpeta de modelos relativa a este archivo
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")

print("Cargando modelos en la GPU... (esto puede tardar un poco)")
# Cargamos los 3 modelos usando los nombres exactos de tu carpeta
MODEL_RESNET = tf.keras.models.load_model(os.path.join(MODELS_DIR, "model_resnet.h5"))
MODEL_CNN = tf.keras.models.load_model(os.path.join(MODELS_DIR, "model_cnn.h5"))
MODEL_MOBILENET = tf.keras.models.load_model(os.path.join(MODELS_DIR, "mobileNet.h5"))

def run_ensemble_prediction(image_file):
    # 1. Leer la imagen original
    img = Image.open(image_file).convert('RGB')

    # 2. Pre-procesamiento para ResNet y CNN (50x50, normalizado 1/255)
    img_50 = img.resize((50, 50))
    img_50_array = np.array(img_50) / 255.0
    img_50_batch = np.expand_dims(img_50_array, axis=0)

    # 3. Pre-procesamiento para MobileNet (96x96, usando preprocess_input de keras)
    img_96 = img.resize((96, 96))
    img_96_array = np.array(img_96)
    img_96_pre = mobilenet_preprocess(img_96_array)
    img_96_batch = np.expand_dims(img_96_pre, axis=0)

    # 4. Obtener predicciones individuales
    p_resnet = MODEL_RESNET.predict(img_50_batch)[0][0]
    p_cnn = MODEL_CNN.predict(img_50_batch)[0][0]
    p_mobilenet = MODEL_MOBILENET.predict(img_96_batch)[0][0]

    # 5. Aplicar lógica de ensamble (Tus pesos: 45% Axel, 45% Gabriel, 10% David)
    # Nota: Axel = ResNet, Gabriel = CNN, David = MobileNet
    prob_final = (p_resnet * 0.45) + (p_cnn * 0.45) + (p_mobilenet * 0.10)
    
    # Usamos tu umbral de 0.40
    umbral = 0.40
    resultado = "Positivo (Cáncer)" if prob_final > umbral else "Negativo (Sano)"

    # 6. Cálculo de "Desconfianza" para el doctor
    # Si la diferencia entre los modelos principales es mucha, avisamos.
    diferencia = abs(p_resnet - p_cnn)
    nivel_desconfianza = "Baja"
    if diferencia > 0.4:
        nivel_desconfianza = "Alta"
    elif diferencia > 0.2:
        nivel_desconfianza = "Media"

    return {
        "clase": 1 if prob_final > umbral else 0,
        "prediccion": resultado,
        "confianza": float(prob_final), # El HTML lo multiplica por 100
        "mensaje": f"Análisis completado. Nivel de desconfianza del sistema: {nivel_desconfianza}. " + 
                   f"Variabilidad diagnóstica: {round(diferencia, 2)}",
        "veredicto": resultado, # Mantenemos estos por si acaso
        "probabilidad_cancer": f"{float(prob_final * 100):.2f}%",
        "confianza_sistema": nivel_desconfianza
    }