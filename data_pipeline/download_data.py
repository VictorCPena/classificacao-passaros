# data_pipeline/01_download_data.py
import requests
import asyncio
import os
import time
from collections import defaultdict
from config import API_URL, QUERY, TARGET_SPECIES_COUNT, MAX_SONGS_PER_SPECIES, DOWNLOADS_DIR

def baixar_arquivo(url, caminho_completo):
    max_retries = 4
    base_delay = 10
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    for attempt in range(max_retries):
        try:
            if attempt > 0: print(f"      Tentativa {attempt + 1}/{max_retries}...")
            response = requests.get(url, stream=True, timeout=30, headers=headers)
            if response.status_code == 200:
                with open(caminho_completo, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
                return True
            elif response.status_code == 503:
                delay = base_delay * (attempt + 1)
                print(f"    Servidor ocupado (Erro 503). Aguardando {delay}s para tentar novamente.")
                time.sleep(delay)
                continue
            else:
                print(f"    Erro HTTP {response.status_code} ao acessar a URL.")
                return False
        except requests.exceptions.RequestException as e:
            print(f"    Erro de rede na tentativa {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(base_delay * (attempt + 1))
    print("    -> Falha final após múltiplas tentativas.")
    return False

async def coletar_e_baixar_especies():
    print(f"--- Coletor de Espécies do Brasil (Alvo: {TARGET_SPECIES_COUNT} espécies) ---")
    
    especies_unicas = set()
    all_recordings = []
    current_page = 1
    num_pages = 1

    print("\nIniciando busca por metadados...")
    while current_page <= num_pages:
        if len(especies_unicas) >= TARGET_SPECIES_COUNT:
            print(f"\nAlvo de {TARGET_SPECIES_COUNT} espécies distintas atingido!")
            break

        print(f"Buscando página {current_page}/{num_pages if num_pages > 1 else '...'}...")
        params = {'query': QUERY, 'page': current_page}
        response = await asyncio.to_thread(requests.get, API_URL, params=params)

        if response.status_code != 200:
            print(f"Erro ao buscar a página {current_page}. Status: {response.status_code}")
            break

        page_metadata = response.json()
        if current_page == 1:
            num_pages = int(page_metadata.get('numPages', 1))
            print(f"Total de páginas a serem buscadas: {num_pages}")

        if not page_metadata.get('recordings'): break
        
        page_recordings = page_metadata['recordings']
        all_recordings.extend(page_recordings)
        for gravacao in page_recordings:
            if gravacao.get('group') == 'birds':
                especies_unicas.add(f"{gravacao['gen']} {gravacao['sp']}")
        
        current_page += 1
        if current_page <= num_pages: await asyncio.sleep(2)

    print(f"\nBusca concluída. {len(all_recordings)} gravações de {len(especies_unicas)} espécies encontradas.")
    
    print("\n--- INICIANDO DOWNLOAD ---")
    gravacoes_por_especie = defaultdict(list)
    for rec in all_recordings:
        if rec.get('group') == 'birds' and rec.get('q', 'C') in ['A', 'B']:
            gravacoes_por_especie[f"{rec['gen']} {rec['sp']}"].append(rec)

    os.makedirs(DOWNLOADS_DIR, exist_ok=True)
    
    total_especies_para_baixar = len(gravacoes_por_especie)
    for i, (especie, gravacoes) in enumerate(gravacoes_por_especie.items()):
        print(f"\n({i+1}/{total_especies_para_baixar}) Processando: {especie}")
        pasta_especie = os.path.join(DOWNLOADS_DIR, especie.replace(" ", "_"))
        os.makedirs(pasta_especie, exist_ok=True)
        
        for j, gravacao in enumerate(gravacoes[:MAX_SONGS_PER_SPECIES]):
            file_url = "https:" + gravacao['file'] if gravacao['file'].startswith('//') else gravacao['file']
            extension = os.path.splitext(gravacao.get('file-name', 'audio.mp3'))[1] or ".mp3"
            file_name = f"XC_{gravacao['id']}{extension}"
            print(f"  ({j+1}/{len(gravacoes[:MAX_SONGS_PER_SPECIES])}) Baixando: {file_name}...")
            sucesso = await asyncio.to_thread(baixar_arquivo, file_url, os.path.join(pasta_especie, file_name))
            print("    -> Sucesso!" if sucesso else "    -> Falha.")
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(coletar_e_baixar_especies())