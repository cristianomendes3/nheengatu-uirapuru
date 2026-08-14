import pandas as pd
import json
import re
from context_extractor import load_ai_ecosystem, extract_contextual_vector

ARQUIVO_ENTRADA = "data/raw/sentencas.csv"
ARQUIVO_SAIDA_JSON = "data/processed/embeddings_extraidos.json"

def clean_context_sentence(text):
    """Pré-processamento não destrutivo: remove apenas espaços extras."""
    if pd.isna(text):
        return ""
    text = str(text).strip()
    return re.sub(r'\s+', ' ', text)

def processar_extracao_contextual():
    print(f"[INFO] Carregando matriz de contexto: {ARQUIVO_ENTRADA}")
    df = pd.read_csv(ARQUIVO_ENTRADA, sep=',')
    df.columns = [col.strip() for col in df.columns]
    
    # Aplica a limpeza estrutural leve
    df['Nheengatu'] = df['Nheengatu'].apply(clean_context_sentence)
    df['Palavra Alvo'] = df['Palavra Alvo'].apply(clean_context_sentence)
    df['Sentença no português'] = df['Sentença no português'].apply(clean_context_sentence)
    df['Tradução'] = df['Tradução'].apply(clean_context_sentence)
        
    tok_yrl, mod_yrl, tok_pt, mod_pt = load_ai_ecosystem()
    resultados = []
    
    print("\n[INFO] Iniciando extração bidirecional do Offset Mapping...")
    for idx, row in df.iterrows():
        oracao_yrl = row['Nheengatu']
        alvo_yrl = row['Palavra Alvo']
        oracao_pt = row['Sentença no português']
        alvo_pt = row['Tradução']
        
        vec_yrl = extract_contextual_vector(oracao_yrl, alvo_yrl, tok_yrl, mod_yrl)
        vec_pt = extract_contextual_vector(oracao_pt, alvo_pt, tok_pt, mod_pt)
        
        # Log de Segurança Crítico (Garante que nenhum tensor nulo passe para a matriz)
        if vec_yrl is None:
            print(f"⚠️ [ALERTA YRL] Falha de rastreio na Linha {idx+2}: Alvo '{alvo_yrl}' não encontrado em '{oracao_yrl}'.")
            continue
        if vec_pt is None:
            print(f"⚠️ [ALERTA PT] Falha de rastreio na Linha {idx+2}: Alvo '{alvo_pt}' não encontrado em '{oracao_pt}'.")
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
        
    print(f"\n[SUCESSO] Extração concluída. {len(resultados)} pares vetoriais salvos em: {ARQUIVO_SAIDA_JSON}")

if __name__ == "__main__":
    processar_extracao_contextual()