import pandas as pd
from datasets import Dataset
import os
import sys
from normalizer import clean_text_nheengatu # Importando o nosso escudo

ARQUIVO_ENTRADA = "data/raw/palavras_nheengatu_completo.xlsx"
ARQUIVO_SAIDA = "data/processed/dataset_nheengatu_raw"

def ingest_and_clean_data():
    print(f"[INFO] Iniciando ingestão do arquivo: {ARQUIVO_ENTRADA}")

    if not os.path.exists(ARQUIVO_ENTRADA):
        print(f"[ERRO] O arquivo {ARQUIVO_ENTRADA} não foi encontrado.")
        sys.exit(1)

    try:
        df = pd.read_excel(ARQUIVO_ENTRADA, engine='openpyxl')
    except Exception as e:
        print(f"[ERRO] Erro ao carregar o arquivo: {e}")
        sys.exit(1)

    if 'Palavra' not in df.columns or 'Significado' not in df.columns:
        print(f"[ERRO] Colunas 'Palavra' e 'Significado' ausentes.")
        sys.exit(1)

    print(f"[INFO] Base carregada. Total bruto: {len(df)} linhas.")

    # APLICAÇÃO DO ESCUDO TECNOLÓGICO (Normalização NFC e Limpeza)
    print("[INFO] Aplicando normalização NFC e limpeza algorítmica...")
    df['Palavra'] = df['Palavra'].apply(clean_text_nheengatu)
    df['Significado'] = df['Significado'].apply(clean_text_nheengatu)
    
    # Removendo possíveis linhas que ficaram vazias após a limpeza
    df = df[(df['Palavra'] != "") & (df['Significado'] != "")]

    # Conversão para Hugging Face Dataset
    print("[INFO] Convertendo para a estrutura Apache Arrow (Hugging Face)...")
    hf_dataset = Dataset.from_pandas(df)
    
    print(f"[SUCESSO] Dataset limpo gerado com {hf_dataset.num_rows} registros estruturados.")
    return hf_dataset

if __name__ == "__main__":
    dataset = ingest_and_clean_data()
    # Exibir primeira linha para auditoria visual
    print(dataset[0])