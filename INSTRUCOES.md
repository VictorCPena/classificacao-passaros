# Guia de Execução do Projeto Classificador de Pássaros

## 1. Introdução

Este documento fornece um guia passo a passo para instalar, configurar e executar todas as etapas do projeto de classificação de cantos de pássaros.

O projeto utiliza um script orquestrador chamado `main.py`, localizado na pasta raiz. Todos os comandos devem ser executados a partir desta pasta para simplificar o processo.

## 2. Pré-requisitos

Antes de começar, garanta que você tenha os seguintes softwares instalados em sua máquina:

* **Python 3.8 ou superior**.
* **Git** para clonar o projeto.
* Um **terminal** ou linha de comando (Terminal no macOS/Linux, PowerShell ou CMD no Windows).

## 3. Passo a Passo da Instalação

Siga estas etapas para configurar o ambiente do projeto.

### Passo 1: Clonar o Repositório

Abra seu terminal, navegue até a pasta onde deseja salvar o projeto e execute o comando abaixo:
```bash
git clone <URL_DO_SEU_REPOSITORIO_GIT>
```
*Substitua `<URL_DO_SEU_REPOSITORIO_GIT>` pela URL correta do seu projeto.*

### Passo 2: Acessar a Pasta do Projeto

Após o download, entre na pasta raiz do projeto:
```bash
cd classificador_passaros
```

### Passo 3: Criar e Ativar um Ambiente Virtual (Recomendado)

Para evitar conflitos com outras bibliotecas do seu sistema, é uma excelente prática criar um ambiente virtual isolado para o projeto.

* **No macOS ou Linux:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

* **No Windows:**
    ```bash
    python -m venv .venv
    .\.venv\Scripts\activate
    ```
*Seu terminal deve agora mostrar `(.venv)` no início da linha, indicando que o ambiente está ativo.*

### Passo 4: Instalar as Dependências

Com o ambiente virtual ativo, instale todas as bibliotecas necessárias com um único comando:
```bash
pip install -r requirements.txt
```
*Este processo pode demorar alguns minutos, pois instalará o TensorFlow e outras bibliotecas.*

## 4. Executando o Projeto

O fluxo de trabalho do projeto é dividido em duas fases principais e independentes: **1. Treinamento** e **2. Serviço (API)**.

Primeiro, você executa o processo de treinamento, que é computacionalmente intensivo e só precisa ser feito uma vez para cada modelo. Este processo gera os arquivos de modelo salvos. Depois, com os modelos já treinados, você pode iniciar o serviço da API a qualquer momento. A API é leve e rápida, pois apenas carrega o modelo pronto para fazer predições.

Todos os comandos são executados a partir da raiz do projeto (`classificador_passaros/`).

### FASE 1: TREINAMENTO (Processo pesado, feito uma única vez por modelo)

**Opção A: Rodar tudo de uma vez (Recomendado para o primeiro uso)**

Este comando executa todo o pipeline de dados e, ao final, treina o modelo especificado.

* **Para gerar tudo e treinar a CNN:**
    ```bash
    python main.py run_all --model cnn
    ```
* **Para gerar tudo e treinar o SVM:**
    ```bash
    python main.py run_all --model svm
    ```
* **Para gerar tudo e treinar o Random Forest:**
    ```bash
    python main.py run_all --model random_forest
    ```

**Opção B: Rodar por Etapas (Para mais controle)**

1.  **Coletar e Preparar Dados:**
    * Este comando baixa os áudios, cria os chunks e gera os espectrogramas.
    ```bash
    python main.py data
    ```

2.  **Extrair Características para Modelos Clássicos:**
    * *(Requer que a Etapa 1 tenha sido executada)*
    * Este comando cria o arquivo `audio_features.csv` com os MFCCs. É necessário **apenas** se você for treinar o SVM ou o Random Forest.
    ```bash
    python main.py extract_features
    ```

3.  **Treinar um Modelo Específico:**
    * *(Requer que as etapas anteriores correspondentes tenham sido executadas)*
    * **Para a CNN:**
        ```bash
        python main.py train --model cnn
        ```
    * **Para o SVM (com otimização):**
        ```bash
        python main.py train --model svm
        ```
    * **Para o Random Forest:**
        ```bash
        python main.py train --model random_forest
        ```

---
### FASE 2: SERVIÇO E PREDIÇÃO (Processo leve, executado a qualquer momento)

Uma vez que a **CNN foi treinada (FASE 1)** e o arquivo `modelo_passaros_finetuned.keras` foi salvo na pasta `service/`, você pode iniciar a API para fazer predições em tempo real.

1.  **Iniciar o Servidor da API:**
    ```bash
    python main.py serve
    ```
    *O terminal ficará ocupado, mostrando que o servidor está no ar em `http://127.0.0.1:8000`. Para parar o servidor, pressione `CTRL+C`.*

2.  **Testar a API:**
    * Abra um **novo terminal** (deixe o servidor rodando no primeiro) e use o comando abaixo para enviar um arquivo de áudio e receber a previsão da espécie.
    * *Substitua `/caminho/para/seu/audio.mp3` por um caminho real de um arquivo de áudio em sua máquina.*
    ```bash
    curl -X POST -F "file=@/caminho/para/seu/audio.mp3" [http://127.0.0.1:8000/predict/](http://127.0.0.1:8000/predict/)
    ```

## 5. Resultados Esperados

Após a execução dos scripts, você verá novos arquivos e pastas sendo criados:
* **Pastas de Dados:** `downloads_xenocanto/`, `dataset_chunks_wav/`, `dataset_espectrogramas/`, `dataset_final_passaros/`.
* **Na pasta `model_pipeline/`:**
    * `audio_features.csv`: O arquivo com os MFCCs.
    * Modelos salvos: `svm_model_tuned.pkl`, `random_forest_model.pkl`.
    * Arquivos de suporte: `scaler.pkl`, `label_encoder.pkl`.
    * Gráficos de análise: `random_forest_feature_importance.png`, `matriz_confusao_cnn.png`.
* **Na pasta `service/`:**
    * O modelo da CNN treinado: `modelo_passaros_finetuned.keras`.
    * O mapeamento de classes: `class_names.json`.