import sys
import os
import pandas as pd
import torch
import json
import warnings

# --- CORREÇÃO DE ROTA ABSOLUTA ---
# Pega o caminho exato de onde você rodou o comando (a raiz nheengatu-uirapuru)
PROJECT_ROOT = os.getcwd()

# Mapeia a pasta models dinamicamente para permitir a importação
sys.path.append(os.path.join(PROJECT_ROOT, "src", "models"))

try:
    from context_extractor import load_ai_ecosystem, extract_contextual_vector
except ModuleNotFoundError:
    print("[ERRO FATAL] Não foi possível encontrar o arquivo 'context_extractor.py' na pasta 'src/models/'.")
    sys.exit(1)

# Silenciar avisos e logs não essenciais
warnings.filterwarnings("ignore")

# Caminhos apontando para a raiz do terminal
ARQUIVO_SENTENCAS = os.path.join(PROJECT_ROOT, "data", "raw", "sentencas.csv")
ARQUIVO_DICIONARIO_JSON = os.path.join(PROJECT_ROOT, "data", "processed", "dataset_nheengatu_expandido.json")
ARQUIVO_DICIONARIO_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "dataset_nheengatu_expandido.csv")

OUT_X_POOL = os.path.join(PROJECT_ROOT, "data", "processed", "pool_X_yrl.pt")
OUT_Y_POOL = os.path.join(PROJECT_ROOT, "data", "processed", "pool_Y_pt.pt")
OUT_VOCAB = os.path.join(PROJECT_ROOT, "data", "processed", "pool_vocab.json")

def garimpar_piscina_contextualizada():
    print(f"[INFO] Iniciando o garimpeiro a partir da raiz: {PROJECT_ROOT}")
    
    # 1. Busca Flexível das Sentenças
    if not os.path.exists(ARQUIVO_SENTENCAS):
        fallback_sentencas = os.path.join(PROJECT_ROOT, "sentencas.csv")
        if os.path.exists(fallback_sentencas):
            caminho_sentencas = fallback_sentencas
        else:
            print(f"[ERRO] Sentenças não encontradas em:\n -> {ARQUIVO_SENTENCAS}\n -> {fallback_sentencas}")
            return
    else:
        caminho_sentencas = ARQUIVO_SENTENCAS

    # 2. Busca Flexível do Dicionário (Aceita JSON ou CSV)
    caminho_dic = None
    eh_json = False
    if os.path.exists(ARQUIVO_DICIONARIO_JSON):
        caminho_dic = ARQUIVO_DICIONARIO_JSON
        eh_json = True
    elif os.path.exists(ARQUIVO_DICIONARIO_CSV):
        caminho_dic = ARQUIVO_DICIONARIO_CSV
    else:
        print(f"[ERRO] Dicionário não encontrado. Procurei por:\n -> {ARQUIVO_DICIONARIO_JSON}\n -> {ARQUIVO_DICIONARIO_CSV}")
        return
        
    print(f"   -> Usando Sentenças de: {caminho_sentencas}")
    print(f"   -> Usando Dicionário de: {caminho_dic}")

    # Carrega os Dados
    df_sentencas = pd.read_csv(caminho_sentencas)
    
    dicionario = []
    if eh_json:
        with open(caminho_dic, 'r', encoding='utf-8') as f:
            dicionario = json.load(f)
    else:
        df_dic = pd.read_csv(caminho_dic)
        for _, row in df_dic.iterrows():
            yrl = row.get("palavra", row.iloc[0])
            pt = row.get("significado", row.iloc[1])
            dicionario.append({"palavra": yrl, "significado": pt})
            
    # Carrega a IA
    tok_yrl, mod_yrl, tok_pt, mod_pt = load_ai_ecosystem()
    
    lista_X, lista_Y, vocabularios = [], [], []
    pares_encontrados = 0
    
    print(f"\n[INFO] Iniciando mineração in-domain de {len(dicionario)} palavras. Isso pode levar alguns segundos...")
    
    for item in dicionario:
        palavra_yrl = str(item.get("palavra", "")).strip()
        palavra_pt = str(item.get("significado", "")).strip()
        
        # Limpa o PT caso haja múltiplos significados (ex: "puranga; poranga")
        palavra_pt = palavra_pt.split(';')[0].split(',')[0].strip()
        
        if not palavra_yrl or not palavra_pt:
            continue
            
        yrl_lower = palavra_yrl.lower()
        pt_lower = palavra_pt.lower()
        
        frase_yrl_encontrada = None
        frase_pt_encontrada = None
        
        # Varredura (Garimpo)
        for _, row in df_sentencas.iterrows():
            s_yrl = str(row['Nheengatu'])
            s_pt = str(row['Sentença no português'])
            
            if yrl_lower in s_yrl.lower() and pt_lower in s_pt.lower():
                frase_yrl_encontrada = s_yrl
                frase_pt_encontrada = s_pt
                break
        
        if frase_yrl_encontrada and frase_pt_encontrada:
            vec_yrl = extract_contextual_vector(frase_yrl_encontrada, palavra_yrl, tok_yrl, mod_yrl)
            vec_pt = extract_contextual_vector(frase_pt_encontrada, palavra_pt, tok_pt, mod_pt)
            
            if vec_yrl is not None and vec_pt is not None:
                lista_X.append(vec_yrl)
                lista_Y.append(vec_pt)
                vocabularios.append({"yrl": palavra_yrl, "pt": palavra_pt})
                pares_encontrados += 1
                
    if pares_encontrados == 0:
        print("❌ Nenhum par contextualizado pôde ser extraído da base de sentenças.")
        return
        
    tensor_X = torch.stack(lista_X)
    tensor_Y = torch.stack(lista_Y)
    
    print(f"\n✅ Garimpo Concluído com Sucesso! {tensor_X.shape[0]} pares com contexto real foram extraídos para a piscina.")
    
    os.makedirs(os.path.dirname(OUT_X_POOL), exist_ok=True)
    torch.save(tensor_X, OUT_X_POOL)
    torch.save(tensor_Y, OUT_Y_POOL)
    with open(OUT_VOCAB, "w", encoding="utf-8") as f:
        json.dump(vocabularios, f, ensure_ascii=False, indent=2)
        
    print("💾 Piscina In-Domain salva com sucesso. O Procrustes Iterativo está pronto para rodar.")

if __name__ == "__main__":
    garimpar_piscina_contextualizada()