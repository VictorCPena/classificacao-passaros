# model_pipeline/evaluate.py
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# --- PARÂMETROS ---
MODEL_PATH = "../service/modelo_passaros_finetuned.keras"
CLASS_NAMES_PATH = "../service/class_names.json"
TEST_DIR = "../dataset_final_passaros/test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# --- VERIFICAÇÃO ---
if not all(os.path.exists(p) for p in [MODEL_PATH, TEST_DIR, CLASS_NAMES_PATH]):
    print("ERRO: Modelo CNN, pasta de teste ou arquivo de classes não encontrado.")
    print("Execute 'python train_unified.py --model cnn' primeiro.")
    exit()

# --- CARREGAMENTO ---
print("Carregando modelo, classes e dados de teste para avaliação da CNN...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)

test_dataset = image_dataset_from_directory(
    TEST_DIR, label_mode='int', image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)

# --- PREDIÇÃO ---
print("Realizando predições no conjunto de teste...")
y_pred_probs = model.predict(test_dataset)
y_pred_labels = np.argmax(y_pred_probs, axis=1)
y_true_labels = np.concatenate([y for x, y in test_dataset], axis=0)

# --- RELATÓRIO DE CLASSIFICAÇÃO ---
print("\n--- Relatório de Classificação da CNN---\n")
print(classification_report(y_true_labels, y_pred_labels, target_names=class_names, zero_division=0))

# --- MATRIZ DE CONFUSÃO ---
print("Gerando matriz de confusão...")
cm = confusion_matrix(y_true_labels, y_pred_labels)
plt.figure(figsize=(15, 15))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusão - CNN')
plt.ylabel('Classe Verdadeira'); plt.xlabel('Classe Prevista')
plt.tight_layout()
plt.savefig("matriz_confusao_cnn.png")
print("\nMatriz de confusão salva como 'matriz_confusao_cnn.png'.")