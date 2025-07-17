import os
import shutil
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import CHUNKS_DIR, SPECTROGRAMS_DIR, N_MELS, FMAX, IMG_DPI

def gerar_espectrograma(caminho_audio, caminho_saida):
    try:
        y, sr = librosa.load(caminho_audio, sr=None)
        if len(y) < 2048: return False 

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=FMAX)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        fig = plt.figure(figsize=[1, 1], dpi=IMG_DPI)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        librosa.display.specshow(S_db, sr=sr, ax=ax, fmax=FMAX)
        plt.savefig(caminho_saida)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"Erro ao gerar espectrograma para {caminho_audio}: {e}")
        return False

def main():
    print(f"--- Gerando espectrogramas dos chunks de áudio ---")
    if not os.path.exists(CHUNKS_DIR):
        print(f"ERRO: A pasta de chunks '{CHUNKS_DIR}' não foi encontrada.")
        return

    if os.path.exists(SPECTROGRAMS_DIR): shutil.rmtree(SPECTROGRAMS_DIR)
    os.makedirs(SPECTROGRAMS_DIR, exist_ok=True)
    
    chunks_a_processar = [os.path.join(r, f) for r, _, fs in os.walk(CHUNKS_DIR) for f in fs if f.lower().endswith('.wav')]

    for chunk_path in tqdm(chunks_a_processar, desc="Gerando Espectrogramas"):
        especie = os.path.basename(os.path.dirname(chunk_path))
        img_dir = os.path.join(SPECTROGRAMS_DIR, especie)
        os.makedirs(img_dir, exist_ok=True)
        img_name = os.path.splitext(os.path.basename(chunk_path))[0] + ".png"
        gerar_espectrograma(chunk_path, os.path.join(img_dir, img_name))

if __name__ == "__main__":
    main()