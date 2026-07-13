import pandas as pd
import json
import re
from transformers import AutoTokenizer
import os
import sys

# Importa a função de ingestão e limpeza do dataset
from ingest_data import ingest_and_clean_data

ARQUIVO_SAIDA_JSON = "data/processed/dataset_nheengatu_expandido.json"
ARQUIVO_SAIDA_CSV = "data/processed/dataset_nheengatu_expandido.csv"
MODELO_NOME = "dominguesm/canarim-bert-nheengatu"

def expandir_dados_cartesianamente(hf_dataset):
    """
    Desmembra entradas polissêmicas (ex: 'Cuára, Kuara' | 'buraco, cova')
    gerando instâncias isoladas para o treinamento da rede neural.
    """
    linhas_expandidas = []
    
    for row in hf_dataset:
        # Divide as strings usando vírgula ou ponto e vírgula como delimitador
        palavras = re.split(r'\s*[,;]\s*', row['Palavra'])
        significados = re.split(r'\s*[,;]\s*', row['Significado'])
        
        # Produto Cartesiano (cruza cada variante de escrita com cada significado)
        for p in palavras:
            for s in significados:
                # Evita linhas vazias causadas por pontuação residual
                if p.strip() and s.strip():
                    linhas_expandidas.append({
                        'Palavra': p.strip(),
                        'Significado': s.strip(),
                        'Raw_Original': row['Palavra'] # Guarda a origem para auditoria
                    })
                    
    df_expandido = pd.DataFrame(linhas_expandidas)
    return df_expandido

def testar_visao_wordpiece(df_expandido, tokenizer):
    """
    Testa se o modelo Canarim-BERT consegue ler as palavras geradas
    sem classificar os morfemas como desconhecidos [UNK].
    """
    print("\n[INFO] Iniciando teste de 'Visão' (Tokenização WordPiece)...")
    
    unk_token_id = tokenizer.unk_token_id
    total_unks = 0
    dados_processados = []
    
    for _, row in df_expandido.iterrows():
        palavra = row['Palavra']
        
        # O tokenizador fatia a palavra (ex: taina-miri -> ['taina', '-', 'miri'])
        tokens = tokenizer.tokenize(palavra)
        input_ids = tokenizer.encode(palavra, add_special_tokens=False)
        
        # Varredura de segurança por [UNK]
        if unk_token_id in input_ids:
            total_unks += 1
            print(f"⚠️ [ALERTA UNK] O modelo não reconheceu: '{palavra}' -> Tokens gerados: {tokens}")
            
        dados_processados.append({
            'palavra': palavra,
            'significado': row['Significado'],
            'tokens': tokens,
            'input_ids': input_ids
        })
        
    print(f"[RESULTADO] Teste de Visão Concluído. Total de falhas [UNK]: {total_unks}")
    return dados_processados

if __name__ == "__main__":
    # 1. Aciona o pipeline de ingestão e limpeza do dataset
    print("--- ETAPA 1: Ingestão e Limpeza ---")
    dataset_limpo = ingest_and_clean_data()
    
    # 2. Expansão Cartesiana
    print("\n--- ETAPA 2: Data Augmentation ---")
    df_expandido = expandir_dados_cartesianamente(dataset_limpo)
    print(f"[SUCESSO] Corpus expandido de {dataset_limpo.num_rows} para {len(df_expandido)} matrizes individuais.")
    
    # 3. Carregamento do Motor de IA
    print(f"\n--- ETAPA 3: Instanciação do Modelo {MODELO_NOME} ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODELO_NOME)
    except Exception as e:
        print(f"[ERRO] Falha ao baixar o tokenizador da nuvem: {e}")
        sys.exit(1)
        
    # 4. Tokenização e Rastreio de Falhas
    dados_finais = testar_visao_wordpiece(df_expandido, tokenizer)
    
    # 5. Exportação dos Dados
    print("\n--- ETAPA 4: Salvamento de Tensores de Segurança ---")
    with open(ARQUIVO_SAIDA_JSON, 'w', encoding='utf-8') as f:
        json.dump(dados_finais, f, ensure_ascii=False, indent=2)
        
    df_expandido.to_csv(ARQUIVO_SAIDA_CSV, index=False, encoding='utf-8-sig', sep=';')
    print(f"[SUCESSO] Dicionário de treino salvo em {ARQUIVO_SAIDA_CSV} e {ARQUIVO_SAIDA_JSON}")