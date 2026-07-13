import json
import torch
import numpy as np
from context_extractor import load_ai_ecosystem, extract_contextual_vector

ARQUIVO_ENTRADA = "data/processed/dataset_nheengatu_expandido.json"
ARQUIVO_SAIDA = "data/processed/embeddings_extraidos.json"

def processar_extracao_massiva():
    print(f"[INFO] Lendo corpus expandido: {ARQUIVO_ENTRADA}")
    with open(ARQUIVO_ENTRADA, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    tok_yrl, mod_yrl, tok_pt, mod_pt = load_ai_ecosystem()
    resultados = []
    
    print("\n[INFO] Iniciando extração de tensores. O Fallback de contexto está ATIVADO.")
    for i, item in enumerate(dataset):
        # Mapeamento Flexível: Tenta puxar o contexto, se não achar, cai para a palavra isolada
        alvo_yrl = item.get('palavra')
        oracao_yrl = item.get('contexto_nheengatu', alvo_yrl) # Fallback 
        
        alvo_pt = item.get('significado')
        oracao_pt = item.get('contexto_portugues', alvo_pt) # Fallback
        
        vec_yrl = extract_contextual_vector(oracao_yrl, alvo_yrl, tok_yrl, mod_yrl)
        vec_pt = extract_contextual_vector(oracao_pt, alvo_pt, tok_pt, mod_pt)
        
        # Tratamento de segurança caso o tokenizador não encontre a palavra na frase
        if vec_yrl is None or vec_pt is None:
            print(f"⚠️ [ALERTA] Falha no Offset Mapping: '{alvo_yrl}' ou '{alvo_pt}'. Pulando.")
            continue
            
        resultados.append({
            "nheengatu_text": alvo_yrl,
            "portuguese_text": alvo_pt,
            "oracao_yrl_utilizada": oracao_yrl,
            "oracao_pt_utilizada": oracao_pt,
            "vetor_yrl": vec_yrl.tolist(),
            "vetor_pt": vec_pt.tolist()
        })
        
        if (i+1) % 50 == 0:
            print(f"  -> Processados {i+1}/{len(dataset)} registros...")
            
    with open(ARQUIVO_SAIDA, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    print(f"\n[SUCESSO] Extração concluída. Matrizes vetoriais salvas em: {ARQUIVO_SAIDA}")

if __name__ == "__main__":
    processar_extracao_massiva()