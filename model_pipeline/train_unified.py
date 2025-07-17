# model_pipeline/train_unified.py
# --- SCRIPT UNIFICADO FINAL COM MLP, SVM E RANDOM FOREST ---

import argparse
import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def train_svm(X_train, X_test, y_train, y_test, label_encoder):
    print("\n--- Otimizando e Treinando Modelo SVM ---")
    param_grid = {'C': [1, 10, 100], 'gamma': ['scale', 'auto', 0.1], 'kernel': ['rbf']}
    grid_search = GridSearchCV(SVC(probability=True, random_state=42), param_grid, cv=5, verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"\nMelhores parâmetros para SVM: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print(f"Acurácia final do SVM no teste: {accuracy_score(y_test, y_pred):.2%}")
    print("\nRelatório de Classificação (SVM):")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))
    joblib.dump(best_model, 'svm_model_tuned.pkl')
    print("\nMelhor modelo SVM salvo como 'svm_model_tuned.pkl'")

def train_random_forest(X_train, X_test, y_train, y_test, label_encoder, feature_names):
    print("\n--- Treinando Modelo Random Forest ---")
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, oob_score=True, class_weight='balanced')
    model.fit(X_train, y_train)
    print(f"\nAcurácia OOB estimada: {model.oob_score_:.2%}")
    y_pred = model.predict(X_test)
    print(f"Acurácia final do Random Forest no teste: {accuracy_score(y_test, y_pred):.2%}")
    print("\nRelatório de Classificação (Random Forest):")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))
    joblib.dump(model, 'random_forest_model.pkl')
    print("\nModelo Random Forest salvo como 'random_forest_model.pkl'")
    
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)
    plt.figure(figsize=(12, 8)); sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20), palette='viridis'); plt.title('Top 20 Características Mais Importantes (Random Forest)'); plt.tight_layout(); plt.savefig("random_forest_feature_importance.png"); plt.close()
    print("Gráfico de importância das características salvo.")



def train_mlp(X_train, X_test, y_train, y_test, label_encoder):
    """
    Treina uma Rede Neural Densa (MLP) com os dados de MFCC.
    """
    if not TF_AVAILABLE:
        print("ERRO: TensorFlow não está instalado. Não é possível treinar a MLP.")
        return

    print("\n--- Treinando Rede Neural Densa (MLP) ---")
    
    n_features = X_train.shape[1]
    n_classes = len(label_encoder.classes_)
    
    model = Sequential([
        Dense(256, activation='relu', input_shape=(n_features,)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(n_classes, activation='softmax') 
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6, verbose=1)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=200,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nAcurácia final da MLP no teste: {accuracy:.2%}")
    
    os.makedirs("../service", exist_ok=True)
    model.save("../service/mlp_model.keras")
    with open("../service/class_names.json", 'w') as f:
        json.dump(label_encoder.classes_.tolist(), f)
    print("Modelo MLP e nomes das classes salvos na pasta 'service/'.")
    
    return history

def main():
    parser = argparse.ArgumentParser(description="Treinador unificado de modelos para classificação de áudio.")
    parser.add_argument('--model', type=str, required=True, choices=['mlp', 'svm', 'random_forest'], help="O modelo a ser treinado.")
    args = parser.parse_args()

    features_file = "audio_features.csv"
    if not os.path.exists(features_file):
        print(f"ERRO: Execute 'extract_features.py' primeiro."); return
        
    df = pd.read_csv(features_file)
    X = df.drop(columns=['species', 'file'])
    y_labels = df['species']
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    if args.model == 'svm':
        train_svm(X_train, X_test, y_train, y_test, label_encoder)
    elif args.model == 'random_forest':
        train_random_forest(X_train, X_test, y_train, y_test, label_encoder, X.columns)
    elif args.model == 'mlp':
        train_mlp(X_train, X_test, y_train, y_test, label_encoder)

if __name__ == "__main__":
    main()
