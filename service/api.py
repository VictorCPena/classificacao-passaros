# service/api.py
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os
from predict import predict_species

app = FastAPI(
    title="API de Classificação de Cantos de Pássaros",
    description="API para classificar espécies da avifauna brasileira a partir de áudio, usando o modelo CNN.",
    version="1.0.0"
)

@app.post("/predict/", summary="Prevê a espécie de um pássaro a partir de um áudio")
async def create_upload_file(file: UploadFile = File(..., description="Arquivo de áudio (.mp3, .wav, .ogg) para classificação.")):
    """
    Recebe um arquivo de áudio, salva-o temporariamente, executa a predição
    usando o modelo CNN treinado e retorna o resultado.
    """
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    # Garante um nome de arquivo único para evitar conflitos
    file_path = os.path.join(temp_dir, f"{os.urandom(8).hex()}_{file.filename}")

    try:
        # Salva o arquivo enviado em um local temporário
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Realiza a predição
        result = predict_species(file_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento do arquivo: {e}")
    finally:
        # Garante que o arquivo temporário seja sempre removido
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/", summary="Endpoint raiz da API")
def read_root():
    """Retorna uma mensagem de boas-vindas e instrução de uso da API."""
    return {"message": "Bem-vindo à API. Use o endpoint /docs para ver a documentação interativa e testar o modelo."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)