API_URL = "https://xeno-canto.org/api/2/recordings"
QUERY = "cnt:Brazil"
TARGET_SPECIES_COUNT = 50 
MAX_SONGS_PER_SPECIES = 4
DOWNLOADS_DIR = "../downloads_xenocanto"

CHUNKS_DIR = "../dataset_chunks_wav"
CHUNK_DURATION_S = 5
TARGET_CHUNKS_PER_SPECIES = 50
NUM_AUGMENTATIONS_PER_CHUNK = 2

SPECTROGRAMS_DIR = "../dataset_espectrogramas"
N_MELS = 128
FMAX = 8000
IMG_DPI = 300

FINAL_DATASET_DIR = "../dataset_final_passaros"
RANDOM_STATE = 42