import os
import torch
import json
import sys

# Resolução dinâmica para importação da biblioteca local
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from csls_filter import compute_csls

# Resolução universal de diretórios baseada no os.getcwd() 
PROJECT_ROOT = os.getcwd()

# Rotas de ancoragem fixa (Treinamento Geométrico)
MATRIZ_X_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "matriz_X_yrl.pt")
MATRIZ_Y_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "matriz_Y_pt.pt")

# Rotas de inteligência linear (Matriz Estrita de Procrustes)
W_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "matriz_W_estrita.pt")

# Rotas de amostragem cega (Piscina In-Domain para teste Zero-Shot)
POOL_X_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "pool_X_yrl.pt")
POOL_Y_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "pool_Y_pt.pt")
VOCAB_POOL_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "pool_vocab.json")

def evaluate_zero_shot():
    """
    Avalia a capacidade de abstração linguística da Matriz de Rotação.

    Carrega a 'Piscina' de tensores isolados e inéditos ao modelo. Centraliza a amostra
    referenciando os vetores de treinamento, aplica o salto topológico pela matriz
    aprendida (W) e quantifica as métricas de aderência semântica (Top-1, Top-3, Top-5)
    do alinhamento não supervisionado.

    Returns:
        None: Exibe relatórios de acurácia global e precisão amostral.
    """
    print("[INFO] Iniciando Pipeline de Testes: Generalização Topológica Zero-Shot...")
    
    # 1. Desserialização de artefatos fixos
    X_anc = torch.load(MATRIZ_X_PATH)
    Y_anc = torch.load(MATRIZ_Y_PATH)
    W = torch.load(W_PATH)
    
    X_test = torch.load(POOL_X_PATH)
    Y_test = torch.load(POOL_Y_PATH)
    
    with open(VOCAB_POOL_PATH, 'r', encoding='utf-8') as f:
        vocab_test = json.load(f)
        
    print(f"[STATUS] Corpus de Validação Inédito Carregado: {X_test.shape[0]} amostras.")
    
    # 2. Centralização rígida orientada aos Baricentros de Treino
    X_train_mean = X_anc.mean(dim=0, keepdim=True)
    Y_train_mean = Y_anc.mean(dim=0, keepdim=True)
    
    X_test_c = X_test - X_train_mean
    
    # 3. Translação e Rotação Ortogonal (Alinhamento Latente)
    Y_pred_c = torch.matmul(X_test_c, W)
    Y_pred = Y_pred_c + Y_train_mean
    
    # 4. Inferência via Vizinhança Penalizada (Hubness Reduction)
    print("[AÇÃO] Mensurando similaridades via cálculo de matrizes CSLS...")
    csls_mat = compute_csls(Y_pred, Y_test, k=5)
    
    # Variáveis de contabilização contínua
    top1_acertos = 0
    top3_acertos = 0
    top5_acertos = 0
    
    total = len(vocab_test)
    
    print("\n--- AMOSTRAGEM GEOMÉTRICA DE TRADUÇÕES INÉDITAS ---")
    print(f"{'Nheengatu (Input)':<20} | {'Previsão IA (W)':<20} | {'Gabarito (Golden)':<20} | {'Precisão'}")
    print("-" * 80)
    
    for i in range(total):
        # Indexação seletiva baseada nas distâncias máximas do filtro topológico
        scores, indices = torch.topk(csls_mat[i], k=5)
        indices = indices.tolist()
        
        src_word = vocab_test[i]['yrl']
        true_word = vocab_test[i]['pt']
        best_pred_word = vocab_test[indices[0]]['pt']
        
        if i == indices[0]: top1_acertos += 1
        if i in indices[:3]: top3_acertos += 1
        if i in indices[:5]: top5_acertos += 1
            
        # Logging de auditoria (Restrito ao 15º par de validação)
        if i < 15:
            status = "🟢 Match Exato (Top-1)" if i == indices[0] else ("🟡 Match Parcial (Top-5)" if i in indices else "🔴 Falha Topológica")
            print(f"{src_word:<20} | {best_pred_word:<20} | {true_word:<20} | {status}")
            
    print("-" * 80)
    print("\n--- PERFORMANCE E AVALIAÇÃO DE CONFIANÇA ---")
    print(f"Taxa de Acurácia Top-1 (Alinhamento Exato) : {(top1_acertos / total) * 100:.2f}% ({top1_acertos}/{total})")
    print(f"Taxa de Acurácia Top-3 (Zona Secundária)   : {(top3_acertos / total) * 100:.2f}% ({top3_acertos}/{total})")
    print(f"Taxa de Acurácia Top-5 (Zona Terciária)    : {(top5_acertos / total) * 100:.2f}% ({top5_acertos}/{total})")
    
    if (top1_acertos / total) > 0.10:
        print("\n[VEREDITO] Aprovação: A matriz W demonstrou habilidade de abstração em vocabulário low-resource.")
    else:
        print("\n[ALERTA] Retenção: O limite probabilístico falhou. Evidência de Overfitting latente à base de treino fixa.")

if __name__ == "__main__":
    evaluate_zero_shot()