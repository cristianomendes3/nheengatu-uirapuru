import pandas as pd
import json
import re
from context_extractor import load_ai_ecosystem, extract_contextual_vector

ARQUIVO_ENTRADA = "data/raw/sentencas.csv"
ARQUIVO_SAIDA_JSON = "data/processed/embeddings_extraidos.json"

def clean_context_sentence(text):
    """
    Pré-processamento brando focado na higienização de espaçamentos.
    Preserva pontuação e estrutura para não danificar o Offset Mapping.

    Args:
        text (str): Texto bruto da sentença.

    Returns:
        str: Texto processado sem espaçamentos residuais duplos.
    """
    if pd.isna(text):
        return ""
    text = str(text).strip()
    return re.sub(r'\s+', ' ', text)

def processar_extracao_contextual():
    """
    Orquestra o pipeline completo de extração de embeddings bilíngues ancorados.

    Consome a planilha de sentenças validadas manualmente, efetua a higienização
    das strings, ativa as redes neurais correspondentes e gera o dicionário
    em formato JSON contendo todos os pares semânticos embutidos em seus vetores densos.

    Returns:
        None: Serializa e salva o arquivo embeddings_extraidos.json em disco.
    """
    print(f"[INFO] Carregando matriz de contexto: {ARQUIVO_ENTRADA}")
    df = pd.read_csv(ARQUIVO_ENTRADA, sep=',')
    df.columns = [col.strip() for col in df.columns]
    
    # Aplicação de limpeza estrutural leve sobre o Dataframe
    df['Nheengatu'] = df['Nheengatu'].apply(clean_context_sentence)
    df['Palavra Alvo'] = df['Palavra Alvo'].apply(clean_context_sentence)
    df['Sentença no português'] = df['Sentença no português'].apply(clean_context_sentence)
    df['Tradução'] = df['Tradução'].apply(clean_context_sentence)
        
    tok_yrl, mod_yrl, tok_pt, mod_pt = load_ai_ecosystem()
    resultados = []
    
    print("\n[INFO] Iniciando extração bidirecional baseada em Offset Mapping...")
    for idx, row in df.iterrows():
        oracao_yrl = row['Nheengatu']
        alvo_yrl = row['Palavra Alvo']
        oracao_pt = row['Sentença no português']
        alvo_pt = row['Tradução']
        
        vec_yrl = extract_contextual_vector(oracao_yrl, alvo_yrl, tok_yrl, mod_yrl)
        vec_pt = extract_contextual_vector(oracao_pt, alvo_pt, tok_pt, mod_pt)
        
        # Filtro de Segurança Crítico: Aborta inserções caso o Mean Pooling falhe em localizar a palavra
        if vec_yrl is None:
            print(f"⚠️ [ALERTA YRL] Falha topológica na Linha {idx+2}: Alvo '{alvo_yrl}' não corresponde à geometria de '{oracao_yrl}'.")
            continue
        if vec_pt is None:
            print(f"⚠️ [ALERTA PT] Falha topológica na Linha {idx+2}: Alvo '{alvo_pt}' não corresponde à geometria de '{oracao_pt}'.")
            continue
            
        resultados.append({
            "linha_origem": idx + 2,
            "oracao_yrl": oracao_yrl,
            "alvo_yrl": alvo_yrl,
            "oracao_pt": oracao_pt,
            "alvo_pt": alvo_pt,
            "vetor_yrl": vec_yrl.tolist(),
            "vetor_pt": vec_pt.tolist()
        })
            
    with open(ARQUIVO_SAIDA_JSON, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
        
    print(f"\n[SUCESSO] Extração concluída. {len(resultados)} pares de representações latentes salvos em: {ARQUIVO_SAIDA_JSON}")

if __name__ == "__main__":
    processar_extracao_contextual()