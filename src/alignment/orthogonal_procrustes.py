import os
import torch
import json
import sys

# Mapeia a pasta alignment para importar o CSLS
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from csls_filter import compute_csls

# Caminhos Inteligentes
PROJECT_ROOT = os.getcwd()
MATRIZ_X_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "matriz_X_yrl.pt")
MATRIZ_Y_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "matriz_Y_pt.pt")
VOCAB_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "embeddings_extraidos.json")
W_OUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "matriz_W_estrita.pt")

def execute_strict_procrustes():
    print("📐 Iniciando Alinhamento Estrito de Procrustes (Closed-Set)...")
    
    # 1. Carrega os tensores puros de 768 dimensões
    X = torch.load(MATRIZ_X_PATH)
    Y = torch.load(MATRIZ_Y_PATH)
    
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
        vocab_yrl = [item.get("alvo_yrl", "") for item in vocab]
        vocab_pt = [item.get("alvo_pt", "") for item in vocab]
        
    print(f"📊 Shape Matriz Nheengatu (X): {list(X.shape)}")
    print(f"📊 Shape Matriz Português (Y): {list(Y.shape)}")
    
    # 2. Centralização Geométrica Estrita
    X_mean = X.mean(dim=0, keepdim=True)
    Y_mean = Y.mean(dim=0, keepdim=True)
    
    X_c = X - X_mean
    Y_c = Y - Y_mean
    
    # 3. Solução Fechada de Procrustes Ortogonal (Schönemann)
    print("🧠 Calculando a SVD e forjando a Matriz de Rotação (W)...")
    M = torch.matmul(X_c.T, Y_c)
    U, S, Vh = torch.linalg.svd(M)
    W = torch.matmul(U, Vh)
    
    # 4. Projeção (Testando o próprio conjunto de treino)
    Y_pred_c = torch.matmul(X_c, W)
    Y_pred = Y_pred_c + Y_mean
    
    # 5. Avaliação Interna (Filtro CSLS)
    print("🔍 Aplicando CSLS para extrair o dicionário limpo...")
    csls_mat = compute_csls(Y_pred, Y, k=5)
    
    best_scores, best_indices = torch.max(csls_mat, dim=1)
    
    acertos = 0
    print("\n🏆 --- TOP 20 PARES DO ALINHAMENTO ESTRITO --- 🏆")
    print(f"{'Nheengatu (Origem)':<20} | {'Português (Alvo)':<20} | {'Score':<6} | {'Status'}")
    print("-" * 75)
    
    for i in range(len(vocab_yrl)):
        src_word = vocab_yrl[i]
        pred_word = vocab_pt[best_indices[i].item()]
        true_word = vocab_pt[i]
        score = best_scores[i].item()
        
        is_correct = (pred_word == true_word)
        if is_correct: 
            acertos += 1
            
        if i < 20: # Mostra os primeiros 20 na tela
            status = "✅" if is_correct else f"❌ (Era: {true_word})"
            print(f"{src_word:<20} | {pred_word:<20} | {score:.4f} | {status}")
            
    print("-" * 75)
    print(f"🎯 Precisão Global da Base Âncora: {(acertos/len(vocab_yrl))*100:.2f}% ({acertos}/{len(vocab_yrl)})")
    
    # 6. Salvar Artefatos
    torch.save(W, W_OUT_PATH)
    print(f"\n💾 Matriz W estrita e centralizada salva com sucesso em: {W_OUT_PATH}")

if __name__ == "__main__":
    execute_strict_procrustes()