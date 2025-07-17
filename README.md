# Classificação de Espécies da Avifauna Brasileira Usando Aprendizado de Máquina

## Resumo

Este trabalho científico apresenta e avalia um pipeline computacional completo para a classificação de espécies de pássaros brasileiros a partir de seus cantos. O projeto compara a performance de três modelos de Machine Learning distintos, todos treinados sobre um vetor de características extraído dos áudios (MFCCs), garantindo uma comparação justa e direta.

## 1. Pipeline de Dados e Racional

Os dados são extraídos da plataforma [Xeno-canto](https://xeno-canto.org/) e passam por um rigoroso processo de tratamento:
1.  **Foco Geográfico:** Gravações do Brasil (`cnt:Brazil`).
2.  **Qualidade do Áudio:** Prioridade para gravações com classificação 'A' e 'B'.
3.  **Feature Engineering:** Os áudios são convertidos em **vetores de MFCCs (Mel-Frequency Cepstral Coefficients)**, que são representações numéricas robustas das características do som.
4.  **Balanceamento e Aumento de Dados:** O pipeline cria múltiplos "chunks" e aplica *Data Augmentation* para aumentar a generalização dos modelos.

## 2. Metodologia e Experimentação de Modelos

1.  **Rede Neural Densa (MLP - Multi-Layer Perceptron):**
    -   **Abordagem Principal.** Representa a abordagem de Deep Learning, usando uma rede neural com múltiplas camadas densas para aprender padrões complexos a partir das características do som.

2.  **Support Vector Machine (SVM):**
    -   **Benchmark Clássico.** Serve como um benchmark de um classificador de margem larga, otimizado com `GridSearchCV`.

3.  **Random Forest (Floresta Aleatória):**
    -   **Benchmark de Ensemble.** Serve como um benchmark de um classificador baseado em *ensemble* de árvores de decisão. Sua vantagem é a **interpretabilidade**, mostrando quais características MFCC foram mais importantes.

## 3. Como Executar

Toda a execução do projeto é centralizada pelo script `main.py`.

**1. Instalação:**
```bash
pip install -r requirements.txt
2. Execução PrincipalPara executar o processo completo (baixar dados, extrair features e treinar TODOS os modelos):python main.py treinar
Este comando é o recomendado para a primeira execução.Outros Comandos (para controle granular)Para preparar os dados e extrair as características (sem treinar):python main.py dados
python main.py features
Para treinar apenas UM modelo específico:# Treinar a Rede Neural MLP
python main.py treinar_um --modelo mlp

# Treinar o SVM
python main.py treinar_um --modelo svm

# Treinar o Random Forest
python main.py treinar_um --modelo random_forest
Para iniciar o serviço da API (com o modelo MLP treinado):python main.py servir
