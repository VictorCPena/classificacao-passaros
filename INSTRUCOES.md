Com certeza\! Preparar um arquivo de instruções claro e separado é uma ótima ideia para compartilhar o projeto.

Crie um arquivo na raiz do seu projeto chamado `INSTRUCOES.md` e cole o conteúdo abaixo nele. O formato Markdown é ideal porque pode ser lido facilmente em qualquer editor de texto e fica com uma ótima formatação em plataformas como GitHub, GitLab, etc.

-----

### **Arquivo: `classificador_passaros/INSTRUCOES.md`**

````markdown
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
git clone https://github.com/VictorCPena/classificacao-passaros
```


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

Agora que tudo está instalado, você pode executar os diferentes fluxos de trabalho do projeto usando o script `main.py`.

### Opção A: Execução Completa (Recomendado para o primeiro uso)

Este comando executa todo o pipeline de uma só vez: baixa os dados, prepara, extrai características e treina o modelo que você escolher.

* **Para rodar tudo e treinar a CNN no final:**
    ```bash
    python main.py run_all --model cnn
    ```
* **Para rodar tudo e treinar o SVM no final:**
    ```bash
    python main.py run_all --model svm
    ```
* **Para rodar tudo e treinar o Random Forest no final:**
    ```bash
    python main.py run_all --model random_forest
    ```

---
### Opção B: Execução por Etapas (Para controle granular)

Se preferir executar cada parte do processo separadamente, siga a ordem abaixo.

* **Etapa 1: Coletar e Preparar Dados**
    * Este comando baixa os áudios do Xeno-canto, cria os chunks e gera os espectrogramas para a CNN.
    ```bash
    python main.py data
    ```

* **Etapa 2: Extrair Características para Modelos Clássicos**
    * *(Requer que a Etapa 1 tenha sido executada)*
    * Este comando cria o arquivo `audio_features.csv` com os MFCCs para o SVM e o Random Forest.
    ```bash
    python main.py extract_features
    ```

* **Etapa 3: Treinar um Modelo Específico**
    * *(Requer que as etapas anteriores correspondentes tenham sido executadas)*
    * **Para treinar a CNN:**
        ```bash
        python main.py train --model cnn
        ```
    * **Para treinar o SVM (com otimização):**
        ```bash
        python main.py train --model svm
        ```
    * **Para treinar o Random Forest:**
        ```bash
        python main.py train --model random_forest
        ```

---
### Opção C: Iniciar a API de Predição

Este comando inicia um servidor web local para que você possa enviar novos áudios e receber a classificação da espécie em tempo real.

* *(Requer que o modelo CNN tenha sido treinado - `python main.py train --model cnn`)*
    ```bash
    python main.py serve
    ```
* O terminal mostrará que o servidor está rodando em `http://127.0.0.1:8000`.
* Para testar, abra **outro terminal** e use o seguinte comando `curl` (substitua pelo caminho de um arquivo de áudio .mp3 ou .wav):
    ```bash
    curl -X POST -F "file=@/caminho/para/seu/audio.mp3" [http://127.0.0.1:8000/predict/](http://127.0.0.1:8000/predict/)
    ```

## 5. Resultados Esperados

Após a execução dos scripts, você verá novos arquivos e pastas sendo criados:
* **Pastas de Dados:** `downloads_xenocanto/`, `dataset_chunks_wav/`, `dataset_espectrogramas/`, `dataset_final_passaros/`.
* **Na pasta `model_pipeline/`:**
    * `audio_features.csv`: O arquivo com os MFCCs.
    * Modelos salvos: `svm_model_tuned.pkl`, `random_forest_model.pkl`.
    * Gráficos de análise: `random_forest_feature_importance.png`.
* **Na pasta `service/`:**
    * O modelo da CNN treinado: `modelo_passaros_finetuned.keras`.
    * O mapeamento de classes: `class_names.json`.
````