# service/predict.py
import tensorflow as tf
import librosa
import numpy as np
import json
import os
import joblib

# --- PARÂMETROS E CARREGAMENTO DE RECURSOS ---
MODEL_PATH = 'mlp_model.keras'
CLASSES_PATH = 'class_names.json'
# O normalizador agora é crucial e precisa ser carregado
SCALER_PATH = '../model_pipeline/scaler.pkl' 

if not all(os.path.exists(p) for p in [MODEL_PATH, CLASSES_PATH, SCALER_PATH]):
    raise FileNotFoundError("Modelo MLP, arquivo de classes ou normalizador não encontrado. Treine o MLP primeiro.")

MODEL = tf.keras.models.load_model(MODEL_PATH)
SCALER = joblib.load(SCALER_PATH)
with open(CLASSES_PATH, 'r') as f:
    CLASS_NAMES = json.load(f)

def extract_features(file_path):
    """Extrai as características MFCC de um arquivo de áudio."""
    try:
        # Carrega o áudio, limitando a 5 segundos para consistência
        y, sr = librosa.load(file_path, sr=None, duration=5) 
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs, axis=1)
        # Retorna como um array 2D para a normalização
        return mfccs_mean.reshape(1, -1) 
    except Exception as e:
        print(f"Erro ao processar {file_path}: {e}")
        return None

def predict_species(audio_path):
    """Executa a predição em um arquivo de áudio usando o modelo MLP."""
    # 1. Extrai as características MFCC do novo áudio
    features = extract_features(audio_path)
    if features is None:
        return {"error": "Não foi possível processar o arquivo de áudio."}
        
    # 2. Normaliza as características usando o MESMO scaler do treino
    scaled_features = SCALER.transform(features)
    
    # 3. Realiza a predição com o modelo MLP
    predictions = MODEL.predict(scaled_features)
    predicted_index = np.argmax(predictions[0])
    
    return {"especie": CLASS_NAMES[predicted_index], "confianca": float(np.max(predictions[0]))}
