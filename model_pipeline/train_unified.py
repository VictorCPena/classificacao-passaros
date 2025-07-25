import argparse
import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance

# Força o Matplotlib a usar um backend não-interativo (Agg)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Tenta importar o TensorFlow e Keras Tuner
try:
    import tensorflow as tf
    import keras_tuner as kt
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# =============================================================================
# SEÇÃO 1: FUNÇÕES DE ANÁLISE GRÁFICA
# =============================================================================

def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    """Gera e salva a matriz de confusão para um modelo."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(18, 18))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Matriz de Confusão - {model_name.upper()}', fontsize=16)
    plt.ylabel('Classe Verdadeira', fontsize=12)
    plt.xlabel('Classe Prevista', fontsize=12)
    plt.tight_layout()
    filename = f"matriz_confusao_{model_name.lower()}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Matriz de confusão salva como '{filename}'")

def plot_training_history(history, model_name):
    """Gera e salva os gráficos de acurácia e perda do treino da MLP."""
    acc = history.history['accuracy']; val_acc = history.history['val_accuracy']
    loss = history.history['loss']; val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1); plt.plot(epochs_range, acc, label='Acurácia de Treino'); plt.plot(epochs_range, val_acc, label='Acurácia de Validação'); plt.legend(loc='lower right'); plt.title('Acurácia de Treino e Validação'); plt.xlabel('Épocas'); plt.ylabel('Acurácia'); plt.grid(True)
    plt.subplot(1, 2, 2); plt.plot(epochs_range, loss, label='Perda de Treino'); plt.plot(epochs_range, val_loss, label='Perda de Validação'); plt.legend(loc='upper right'); plt.title('Perda de Treino e Validação'); plt.xlabel('Épocas'); plt.ylabel('Perda'); plt.grid(True)
    plt.suptitle(f'Histórico de Treinamento - {model_name.upper()}', fontsize=16); plt.tight_layout(rect=[0, 0, 1, 0.96]);
    filename = f"{model_name.lower()}_training_history.png"
    plt.savefig(filename); plt.close()
    print(f"Gráficos do histórico de treino salvos como '{filename}'")

def plot_feature_importance(importance, names, model_name):
    """Gera e salva o gráfico de importância das características."""
    feature_importance_df = pd.DataFrame({'feature': names, 'importance': importance}).sort_values(by='importance', ascending=False)
    
    print(f"\nTop 10 Características Mais Importantes ({model_name.upper()}):")
    print(feature_importance_df.head(10))
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20), palette='viridis')
    plt.title(f'Top 20 Características Mais Importantes ({model_name.upper()})')
    plt.tight_layout()
    filename = f"{model_name.lower()}_feature_importance.png"
    plt.savefig(filename)
    plt.close()
    print(f"Gráfico de importância das características salvo como '{filename}'")

def plot_learning_curve(estimator, X, y, model_name):
    """Gera e salva a curva de aprendizado baseada no tamanho do dataset."""
    print(f"\nGerando curva de aprendizado para o modelo {model_name.upper()}...")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(.1, 1.0, 5), scoring="accuracy")

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure()
    plt.title(f"Curva de Aprendizado - {model_name.upper()}")
    plt.xlabel("Tamanho do Conjunto de Treino")
    plt.ylabel("Acurácia")
    plt.grid()
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Acurácia de Treino")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Acurácia de Validação (Cross-Validation)")
    plt.legend(loc="best")
    
    filename = f"{model_name.lower()}_learning_curve.png"
    plt.savefig(filename)
    plt.close()
    print(f"Curva de aprendizado salva como '{filename}'")


# =============================================================================
# SEÇÃO 2: FUNÇÕES DE TREINAMENTO DOS MODELOS
# =============================================================================

def train_svm(X_train, X_test, y_train, y_test, label_encoder, feature_names):
    """Otimiza e treina um modelo SVM, gerando todas as análises."""
    print("\n--- Otimizando e Treinando Modelo SVM ---")
    param_grid = {'C': [10, 100], 'gamma': ['scale', 'auto'], 'kernel': ['rbf']}
    grid_search = GridSearchCV(SVC(probability=True, random_state=42), param_grid, cv=3, verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"\nMelhores parâmetros para SVM: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    
    y_pred = best_model.predict(X_test)
    print(f"Acurácia final do SVM no teste (validação): {accuracy_score(y_test, y_pred):.2%}")
    print("\nRelatório de Classificação (SVM):")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))
    
    plot_confusion_matrix(y_test, y_pred, label_encoder.classes_, "SVM")
    plot_learning_curve(best_model, X_train, y_train, "SVM")
    
    print("\nCalculando a importância das características para o SVM (Permutation Importance)...")
    perm_importance = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    plot_feature_importance(perm_importance.importances_mean, feature_names, "SVM")
    
    joblib.dump(best_model, 'svm_model_tuned.pkl')
    print("\nMelhor modelo SVM salvo como 'svm_model_tuned.pkl'")

def train_random_forest(X_train, X_test, y_train, y_test, label_encoder, feature_names):
    """Otimiza e treina um modelo Random Forest, gerando todas as análises."""
    print("\n--- Otimizando e Treinando Modelo Random Forest ---")
    
    param_grid = {
        'n_estimators': [200, 300], 
        'max_depth': [10, 20, 30],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4]
    }
    
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'), param_grid, cv=3, verbose=2, n_jobs=-1)
    
    print("Iniciando a busca pelos melhores hiperparâmetros para o Random Forest...")
    grid_search.fit(X_train, y_train)
    
    print(f"\nMelhores parâmetros para Random Forest: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    
    y_pred = best_model.predict(X_test)
    print(f"Acurácia final do Random Forest no teste (validação): {accuracy_score(y_test, y_pred):.2%}")
    print("\nRelatório de Classificação (Random Forest):")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))
    
    plot_confusion_matrix(y_test, y_pred, label_encoder.classes_, "Random_Forest")
    plot_learning_curve(best_model, X_train, y_train, "Random_Forest")
    plot_feature_importance(best_model.feature_importances_, feature_names, "Random_Forest")
    
    joblib.dump(best_model, 'random_forest_model_tuned.pkl')
    print("\nMelhor modelo Random Forest salvo como 'random_forest_model_tuned.pkl'")

def train_mlp(X_train, X_test, y_train, y_test, label_encoder, feature_names):
    """Otimiza e treina uma Rede Neural Densa (MLP), gerando todas as análises."""
    if not TF_AVAILABLE: print("ERRO: TensorFlow e Keras Tuner não estão instalados."); return

    print("\n--- Otimizando e Treinando Rede Neural Densa (MLP) ---")
    
    n_features = X_train.shape[1]
    n_classes = len(label_encoder.classes_)

    def build_model(hp):
        model = Sequential()
        model.add(Dense(units=hp.Int('units_1', 128, 512, 128), activation='relu', input_shape=(n_features,)))
        model.add(BatchNormalization()); model.add(Dropout(rate=hp.Float('dropout_1', 0.3, 0.6, 0.1)))
        model.add(Dense(units=hp.Int('units_2', 64, 256, 64), activation='relu'))
        model.add(BatchNormalization()); model.add(Dropout(rate=hp.Float('dropout_2', 0.3, 0.6, 0.1)))
        model.add(Dense(n_classes, activation='softmax'))
        model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-3, 5e-4])), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    tuner = kt.Hyperband(build_model, objective='val_accuracy', max_epochs=100, factor=3, directory='keras_tuner_dir', project_name='bird_sound_classification')
    stop_early = EarlyStopping(monitor='val_loss', patience=15)
    
    print("Iniciando a busca pelos melhores hiperparâmetros para a MLP...")
    tuner.search(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[stop_early])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"\nMelhores hiperparâmetros encontrados para MLP: {best_hps.values}")

    print("\nTreinando o melhor modelo MLP...")
    best_model = tuner.hypermodel.build(best_hps)
    history = best_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=32, callbacks=[stop_early])
    
    plot_training_history(history, "MLP")
    
    loss, accuracy = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"\nAcurácia final da MLP no teste (validação): {accuracy:.2%}")
    
    y_pred_probs = best_model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    print("\nRelatório de Classificação (MLP):")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))
    
    plot_confusion_matrix(y_test, y_pred, label_encoder.classes_, "MLP")
    
    # --- ALTERAÇÃO AQUI ---
    # A função permutation_importance do scikit-learn não é diretamente compatível com modelos Keras.
    # O gráfico de importância do Random Forest já nos dá uma excelente análise das features.
    # Por isso, a análise de features para a MLP foi removida para evitar o erro.
    print("\nAnálise de importância de características para MLP não será executada (incompatibilidade de biblioteca).")
    
    os.makedirs("../service", exist_ok=True)
    best_model.save("../service/mlp_model.keras")
    with open("../service/class_names.json", 'w') as f: json.dump(label_encoder.classes_.tolist(), f)
    print("Melhor modelo MLP e nomes das classes salvos na pasta 'service/'.")

# =============================================================================
# FUNÇÃO PRINCIPAL (MAIN)
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Treinador unificado de modelos para classificação de áudio.")
    parser.add_argument('--model', type=str, required=True, choices=['mlp', 'svm', 'random_forest'], help="O modelo a ser treinado.")
    args = parser.parse_args()

    features_file = "audio_features.csv"
    if not os.path.exists(features_file): print(f"ERRO: Execute 'extract_features.py' primeiro."); return
        
    df = pd.read_csv(features_file)
    X = df.drop(columns=['species', 'file'])
    y_labels = df['species']
    
    label_encoder = LabelEncoder(); y = label_encoder.fit_transform(y_labels)
    scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
    
    joblib.dump(scaler, 'scaler.pkl'); joblib.dump(label_encoder, 'label_encoder.pkl')

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    if args.model == 'svm':
        train_svm(X_train, X_test, y_train, y_test, label_encoder, X.columns)
    elif args.model == 'random_forest':
        train_random_forest(X_train, X_test, y_train, y_test, label_encoder, X.columns)
    elif args.model == 'mlp':
        train_mlp(X_train, X_test, y_train, y_test, label_encoder, X.columns)

if __name__ == "__main__":
    main()