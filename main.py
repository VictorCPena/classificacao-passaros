# main.py
import argparse
import subprocess
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()

def run_command(command, cwd):
    cwd_str = str(cwd)
    print(f"\n[ORQUESTRADOR] Executando comando: '{' '.join(command)}' em '{cwd_str}'")
    try:
        process = subprocess.run([sys.executable] + command, cwd=cwd_str, check=True, text=True)
    except Exception as e:
        print(f"ERRO ao executar '{' '.join(command)}': {e}")
        sys.exit(1)

def run_data_pipeline():
    print(">>> INICIANDO PIPELINE DE DADOS <<<")
    data_pipeline_dir = PROJECT_ROOT / "data_pipeline"
    scripts = ["download_data.py", "preprocess_audio.py"]
    for script in scripts:
        run_command([script], cwd=data_pipeline_dir)
    print("\n>>> PIPELINE DE DADOS CONCLUÍDO <<<")

def run_feature_extraction():
    print(">>> INICIANDO EXTRAÇÃO DE CARACTERÍSTICAS (MFCCs) <<<")
    model_pipeline_dir = PROJECT_ROOT / "model_pipeline"
    run_command(["extract_features.py"], cwd=model_pipeline_dir)
    print("\n>>> EXTRAÇÃO DE CARACTERÍSTICAS CONCLUÍDA <<<")

def run_training(model_name):
    print(f">>> INICIANDO TREINAMENTO DO MODELO: {model_name.upper()} <<<")
    model_pipeline_dir = PROJECT_ROOT / "model_pipeline"
    run_command(["train_unified.py", "--model", model_name], cwd=model_pipeline_dir)
    print(f"\n>>> TREINAMENTO DO MODELO {model_name.upper()} CONCLUÍDO <<<")

def start_service():
    print(">>> INICIANDO SERVIÇO DA API (FastAPI com Uvicorn) <<<")
    print("Use CTRL+C para parar o servidor.")
    service_dir = PROJECT_ROOT / "service"
    command = ["uvicorn", "api:app", "--reload"]
    try:
        subprocess.run(command, cwd=str(service_dir), check=True)
    except KeyboardInterrupt:
        print("\nServidor da API finalizado.")

def main():
    parser = argparse.ArgumentParser(description="Orquestrador do projeto de classificação de pássaros.", formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True, help="Comandos disponíveis")

    parser_treinar = subparsers.add_parser("treinar", help="Executa o pipeline de dados, extrai features e treina TODOS os 3 modelos.")
    def run_train_all(args):
        run_data_pipeline()
        run_feature_extraction()
        run_training('mlp')
        run_training('svm')
        run_training('random_forest')
        print("\n\n>>> TODOS OS MODELOS FORAM TREINADOS COM SUCESSO! <<<")
    parser_treinar.set_defaults(func=run_train_all)
    
    subparsers.add_parser("dados", help="Executa apenas o pipeline de download e pré-processamento.").set_defaults(func=lambda args: run_data_pipeline())
    subparsers.add_parser("features", help="Executa apenas a extração de características (MFCCs).").set_defaults(func=lambda args: run_feature_extraction())
    
    parser_train_single = subparsers.add_parser("treinar_um", help="Treina apenas UM dos modelos disponíveis.")
    parser_train_single.add_argument("--modelo", type=str, required=True, choices=['mlp', 'svm', 'random_forest'], help="O modelo a ser treinado.")
    parser_train_single.set_defaults(func=lambda args: run_training(args.modelo))

    subparsers.add_parser("servir", help="Inicia a API para servir o modelo MLP treinado.").set_defaults(func=lambda args: start_service())

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()