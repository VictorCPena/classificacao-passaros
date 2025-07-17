# data_pipeline/02_preprocess_audio.py
import os
import shutil
import librosa
import soundfile as sf
import random
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
from tqdm import tqdm
from config import DOWNLOADS_DIR, CHUNKS_DIR, CHUNK_DURATION_S, TARGET_CHUNKS_PER_SPECIES, NUM_AUGMENTATIONS_PER_CHUNK

def chunk_and_augment_audios(input_dir, output_dir, chunk_duration, target_chunks, num_augmentations):
    print(f"--- Cortando e Aumentando Áudios ---")
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    augmenter = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    ])

    species_list = [s for s in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, s))]

    for especie in tqdm(species_list, desc="Processando espécies"):
        path_especie = os.path.join(input_dir, especie)
        audio_files = [os.path.join(path_especie, f) for f in os.listdir(path_especie) if f.lower().endswith(('.mp3', '.wav', '.ogg'))]
        if not audio_files: continue

        species_chunk_dir = os.path.join(output_dir, especie)
        os.makedirs(species_chunk_dir, exist_ok=True)
        chunks_count = 0
        
        while chunks_count < target_chunks:
            audio_path = random.choice(audio_files)
            try:
                y, sr = librosa.load(audio_path, sr=None)
                chunk_samples = int(chunk_duration * sr)
                if len(y) < chunk_samples: continue

                start_sample = random.randint(0, len(y) - chunk_samples)
                chunk_data = y[start_sample : start_sample + chunk_samples]
                original_filename = os.path.splitext(os.path.basename(audio_path))[0]
                
                for i in range(1 + num_augmentations):
                    if chunks_count >= target_chunks: break
                    
                    data_to_save = augmenter(samples=chunk_data, sample_rate=sr) if i > 0 else chunk_data
                    aug_suffix = f"_aug{i-1}" if i > 0 else ""
                    chunk_filename = f"{original_filename}_chunk{chunks_count}{aug_suffix}.wav"
                    
                    sf.write(os.path.join(species_chunk_dir, chunk_filename), data_to_save, sr)
                    chunks_count += 1
            except Exception as e:
                print(f"\nErro ao processar {audio_path}: {e}")

if __name__ == "__main__":
    if not os.path.exists(DOWNLOADS_DIR):
        print(f"ERRO: A pasta de downloads '{DOWNLOADS_DIR}' não foi encontrada.")
    else:
        chunk_and_augment_audios(DOWNLOADS_DIR, CHUNKS_DIR, CHUNK_DURATION_S, TARGET_CHUNKS_PER_SPECIES, NUM_AUGMENTATIONS_PER_CHUNK)