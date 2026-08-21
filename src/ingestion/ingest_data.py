import pandas as pd
from datasets import Dataset
import os
import sys

# Importa o módulo de higienização de texto
from normalizer import clean_text_nheengatu

ARQUIVO_ENTRADA = "data/raw/palavras_nheengatu_completo.xlsx"
ARQUIVO_SAIDA = "data/processed/dataset_nheengatu_raw"

def ingest_and_clean_data():
    """
    Realiza a ingestão, limpeza e normalização do corpus bruto.

    Lê a planilha Excel original, garante a integridade das colunas alvo
    ('Palavra' e 'Significado'), aplica o pipeline de normalização textual
    (NFC e remoção de ruídos) e converte o resultado para a estrutura
    Dataset do Apache Arrow (Hugging Face).

    Returns:
        Dataset: Objeto Dataset contendo os pares semânticos limpos.
    """
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

    # Aplicação do pipeline de sanitização (NFC e Regex)
    print("[INFO] Aplicando normalização NFC e limpeza algorítmica...")
    df['Palavra'] = df['Palavra'].apply(clean_text_nheengatu)
    df['Significado'] = df['Significado'].apply(clean_text_nheengatu)
    
    # Remoção de instâncias nulas residuais pós-limpeza
    df = df[(df['Palavra'] != "") & (df['Significado'] != "")]

    # Estruturação para o padrão de leitura dos modelos Transformers
    print("[INFO] Convertendo para a estrutura Apache Arrow (Hugging Face)...")
    hf_dataset = Dataset.from_pandas(df)
    
    print(f"[SUCESSO] Dataset limpo gerado com {hf_dataset.num_rows} registros estruturados.")
    return hf_dataset

if __name__ == "__main__":
    dataset = ingest_and_clean_data()
    # Log de auditoria visual da primeira entrada processada
    print(dataset[0])