# data_pipeline/04_split_dataset.py
import os
import pandas as pd
import shutil
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm
import random
from config import SPECTROGRAMS_DIR, FINAL_DATASET_DIR, RANDOM_STATE

def organizar_arquivos(file_list, pasta_destino):
    if not file_list: return
    if os.path.exists(pasta_destino): shutil.rmtree(pasta_destino)
    os.makedirs(pasta_destino, exist_ok=True)
    for f_path in tqdm(file_list, desc=f"Copiando para {os.path.basename(pasta_destino)}"):
        especie = os.path.basename(os.path.dirname(f_path))
        especie_dir = os.path.join(pasta_destino, especie)
        os.makedirs(especie_dir, exist_ok=True)
        shutil.copy(f_path, especie_dir)

def main():
    print(f"--- Dividindo dados de forma balanceada (Stratified Group K-Fold) ---")
    if not os.path.exists(SPECTROGRAMS_DIR):
        print(f"ERRO: A pasta de espectrogramas '{SPECTROGRAMS_DIR}' não foi encontrada.")
        return

    mapa_dados = []
    for r, _, fs in os.walk(SPECTROGRAMS_DIR):
        for f in fs:
            if f.lower().endswith('.png'):
                mapa_dados.append({
                    "caminho_imagem": os.path.join(r, f),
                    "especie": os.path.basename(r),
                    "grupo_gravacao": f.split('_chunk')[0]
                })
    df = pd.DataFrame(mapa_dados)
    if df.empty:
        print("Nenhum espectrograma encontrado para dividir.")
        return

    train_files, test_files = [], []
    
    for especie in tqdm(df['especie'].unique(), desc="Dividindo espécies"):
        df_especie = df[df['especie'] == especie]
        grupos_unicos = df_especie['grupo_gravacao'].unique()
        

        if len(grupos_unicos) < 2:
            imagens_especie = df_especie['caminho_imagem'].tolist()
            random.shuffle(imagens_especie)
            ponto_divisao = int(len(imagens_especie) * 0.8)
            train_files.extend(imagens_especie[:ponto_divisao])
            test_files.extend(imagens_especie[ponto_divisao:])
        else:
            n_splits = min(5, len(grupos_unicos))
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
            train_idx, test_idx = next(sgkf.split(df_especie, df_especie['especie'], df_especie['grupo_gravacao']))
            train_files.extend(df_especie.iloc[train_idx]['caminho_imagem'].tolist())
            test_files.extend(df_especie.iloc[test_idx]['caminho_imagem'].tolist())
    
    organizar_arquivos(train_files, os.path.join(FINAL_DATASET_DIR, "train"))
    organizar_arquivos(test_files, os.path.join(FINAL_DATASET_DIR, "test"))
    
    print(f"\nDivisão Concluída: {len(train_files)} arquivos de treino, {len(test_files)} arquivos de teste.")

if __name__ == "__main__":
    main()