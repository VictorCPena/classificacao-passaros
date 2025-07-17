# model_pipeline/extract_features.py
import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

CHUNKS_DIR = "../dataset_chunks_wav"
OUTPUT_CSV_FILE = "audio_features.csv" 

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs, axis=1)
        return mfccs_mean
    except Exception as e:
        print(f"Erro ao processar {file_path}: {e}")
        return None

def main():
    if not os.path.exists(CHUNKS_DIR):
        print(f"ERRO: Diretório de chunks '{CHUNKS_DIR}' não encontrado.")
        print("Execute o pipeline de dados (etapas 01 e 02) primeiro.")
        return

    print(f"--- Iniciando extração de características (MFCCs) de '{CHUNKS_DIR}' ---")
    
    all_features = []
    
    for species_name in tqdm(os.listdir(CHUNKS_DIR), desc="Processando Espécies"):
        species_dir = os.path.join(CHUNKS_DIR, species_name)
        if not os.path.isdir(species_dir): continue
            
        for file_name in os.listdir(species_dir):
            if file_name.lower().endswith('.wav'):
                file_path = os.path.join(species_dir, file_name)
                features = extract_features(file_path)
                
                if features is not None:
                    feature_data = {'species': species_name, 'file': file_name}
                    for i, feat in enumerate(features):
                        feature_data[f'mfcc_{i+1}'] = feat
                    all_features.append(feature_data)

    if not all_features:
        print("Nenhuma característica foi extraída.")
        return

    features_df = pd.DataFrame(all_features)
    output_path = os.path.join(os.path.dirname(__file__), OUTPUT_CSV_FILE)
    features_df.to_csv(output_path, index=False)
    
    print(f"\n--- Extração concluída! ---")
    print(f"Total de {len(features_df)} amostras processadas.")
    print(f"Arquivo de características salvo em: '{output_path}'")

if __name__ == "__main__":
    main()