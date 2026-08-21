import pandas as pd
import json
import re
from transformers import AutoTokenizer
import os
import sys

# Integração do módulo primário de higienização
from ingest_data import ingest_and_clean_data

ARQUIVO_SAIDA_JSON = "data/processed/dataset_nheengatu_expandido.json"
ARQUIVO_SAIDA_CSV = "data/processed/dataset_nheengatu_expandido.csv"
MODELO_NOME = "dominguesm/canarim-bert-nheengatu"

def expandir_dados_cartesianamente(hf_dataset):
    """
    Desmembra blocos semânticos polissêmicos em instâncias de pareamento 1:1.

    Realiza uma combinação de Produto Cartesiano para registros que contêm
    múltiplas ocorrências agrupadas por delimitadores literais (ex: 'buraco, cova').
    Essencial para garantir que a rede neural receba tensores não corrompidos
    por amálgamas textuais.

    Args:
        hf_dataset (Dataset): Objeto Hugging Face Dataset gerado pela ingestão bruta.

    Returns:
        pd.DataFrame: Conjunto expandido e pronto para injeção hiperdimensional.
    """
    linhas_expandidas = []
    
    for row in hf_dataset:
        # Segmentação das matrizes polissêmicas
        palavras = re.split(r'\s*[,;]\s*', row['Palavra'])
        significados = re.split(r'\s*[,;]\s*', row['Significado'])
        
        # Cruzamento combinatório (Produto Cartesiano)
        for p in palavras:
            for s in significados:
                # Prevenção contra instâncias nulas geradas por falhas de segmentação
                if p.strip() and s.strip():
                    linhas_expandidas.append({
                        'Palavra': p.strip(),
                        'Significado': s.strip(),
                        'Raw_Original': row['Palavra']  # Preservação de rastreabilidade
                    })
                    
    df_expandido = pd.DataFrame(linhas_expandidas)
    return df_expandido

def testar_visao_wordpiece(df_expandido, tokenizer):
    """
    Realiza teste de integridade na tokenização do modelo pré-treinado.

    Mapeia e valida se o algoritmo de WordPiece do Transformer alvo consegue
    interpretar os grafemas sem reduzi-los ao token de ignorância ([UNK]),
    avaliando a cobertura lexical do corpus processado.

    Args:
        df_expandido (pd.DataFrame): Dados pós-Data Augmentation.
        tokenizer (AutoTokenizer): Instância do tokenizador Hugging Face.

    Returns:
        list[dict]: Array contendo o dicionário com metadados detalhados de tokenização.
    """
    print("\n[INFO] Iniciando varredura topológica de tokenização (WordPiece)...")
    
    unk_token_id = tokenizer.unk_token_id
    total_unks = 0
    dados_processados = []
    
    for _, row in df_expandido.iterrows():
        palavra = row['Palavra']
        
        # Segmentação morfológica (Ex: taina-miri -> ['taina', '-', 'miri'])
        tokens = tokenizer.tokenize(palavra)
        input_ids = tokenizer.encode(palavra, add_special_tokens=False)
        
        # Rastreio condicional de anomalias dimensionais ([UNK])
        if unk_token_id in input_ids:
            total_unks += 1
            print(f"[ALERTA UNK] O modelo falhou em mapear: '{palavra}' -> Tokens: {tokens}")
            
        dados_processados.append({
            'palavra': palavra,
            'significado': row['Significado'],
            'tokens': tokens,
            'input_ids': input_ids
        })
        
    print(f"[INFO] Diagnóstico concluído. Ocorrências [UNK] isoladas: {total_unks}")
    return dados_processados

if __name__ == "__main__":
    # 1. Acionamento do pipeline raiz de ingestão
    print("--- FASE 1: Extração e Sanitização ---")
    dataset_limpo = ingest_and_clean_data()
    
    # 2. Execução do Data Augmentation (Expansão Cartesiana)
    print("\n--- FASE 2: Expansão Combinatória ---")
    df_expandido = expandir_dados_cartesianamente(dataset_limpo)
    print(f"[SUCESSO] Corpus enriquecido de {dataset_limpo.num_rows} para {len(df_expandido)} registros.")
    
    # 3. Requisição de dependências de Inteligência Artificial
    print(f"\n--- FASE 3: Instanciamento de Componentes Neurais ({MODELO_NOME}) ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODELO_NOME)
    except Exception as e:
        print(f"[ERRO FATAL] Falha de comunicação com o repositório Hugging Face: {e}")
        sys.exit(1)
        
    # 4. Avaliação de integridade lexical
    dados_finais = testar_visao_wordpiece(df_expandido, tokenizer)
    
    # 5. Exportação e serialização de dados limpos
    print("\n--- FASE 4: Serialização de Artefatos ---")
    with open(ARQUIVO_SAIDA_JSON, 'w', encoding='utf-8') as f:
        json.dump(dados_finais, f, ensure_ascii=False, indent=2)
        
    df_expandido.to_csv(ARQUIVO_SAIDA_CSV, index=False, encoding='utf-8-sig', sep=';')
    print(f"[SUCESSO] Base de conhecimento compilada com segurança nos formatos CSV e JSON.")