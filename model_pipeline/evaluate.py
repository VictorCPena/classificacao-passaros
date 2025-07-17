import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

MODEL_PATH = "../service/mlp_model.keras"
CLASS_NAMES_PATH = "../service/class_names.json"
FEATURES_PATH = "audio_features.csv"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"

def main():
    """
    Carrega o modelo MLP treinado e os dados de teste para realizar uma avaliação detalhada,
    incluindo um relatório de classificação e uma matriz de confusão.
    """
    print("--- Iniciando Avaliação Detalhada do Modelo MLP ---")

    required_files = [MODEL_PATH, CLASS_NAMES_PATH, FEATURES_PATH, SCALER_PATH, ENCODER_PATH]
    if not all(os.path.exists(p) for p in required_files):
        print("ERRO: Um ou mais arquivos necessários não foram encontrados.")
        print("Certifique-se de que o modelo MLP foi treinado ('python main.py treinar_um --modelo mlp').")
        return
    
    if not TF_AVAILABLE:
        print("ERRO: TensorFlow não está instalado. Não é possível carregar o modelo.")
        return

    print("Carregando modelo, dados e pré-processadores...")
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    df = pd.read_csv(FEATURES_PATH)

    X = df.drop(columns=['species', 'file'])
    y_labels = df['species']
    
    y = label_encoder.transform(y_labels)
    
    X_scaled = scaler.transform(X)
    
    _, X_test, _, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    print("Realizando predições no conjunto de teste...")
    y_pred_probs = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)

    print("\n--- Relatório de Classificação da MLP ---\n")
    print(classification_report(y_test, y_pred_labels, target_names=label_encoder.classes_, zero_division=0))

    print("Gerando matriz de confusão...")
    cm = confusion_matrix(y_test, y_pred_labels)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Matriz de Confusão - MLP')
    plt.ylabel('Classe Verdadeira')
    plt.xlabel('Classe Prevista')
    plt.tight_layout()
    plt.savefig("matriz_confusao_mlp.png")
    print("\nMatriz de confusão salva como 'matriz_confusao_mlp.png'.")

if __name__ == "__main__":
    main()
