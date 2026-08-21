import os
import torch
import json
import sys

# Resolução dinâmica para importação da biblioteca CSLS
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from csls_filter import compute_csls

# Resolução universal de diretórios
PROJECT_ROOT = os.getcwd()
MATRIZ_X_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "matriz_X_yrl.pt")
MATRIZ_Y_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "matriz_Y_pt.pt")
VOCAB_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "embeddings_extraidos.json")
W_OUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "matriz_W_estrita.pt")

def execute_strict_procrustes():
    """
    Soluciona o Problema Ortogonal de Procrustes para Alinhamento Bilíngue.

    Centraliza as coordenadas nativas de cada idioma subtraindo seus centros 
    de gravidade e computa, através da Decomposição em Valores Singulares (SVD),
    a Matriz W de Transformação Ortogonal (Solução de Schönemann). Esta matriz 
    garante a isometria espacial (não deforma o volume ou ângulos entre as palavras).

    Returns:
        None: Serializa e salva a Matriz de Rotação (matriz_W_estrita.pt) em disco.
    """
    print("[INFO] Iniciando Cômputo do Problema Ortogonal de Procrustes...")
    
    # 1. Carregamento do Espaço Latente Extraído
    X = torch.load(MATRIZ_X_PATH)
    Y = torch.load(MATRIZ_Y_PATH)
    
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
        vocab_yrl = [item.get("alvo_yrl", "") for item in vocab]
        vocab_pt = [item.get("alvo_pt", "") for item in vocab]
        
    print(f"Formato da Camada Oculta Nheengatu (X): {list(X.shape)}")
    print(f"Formato da Camada Oculta Português (Y): {list(Y.shape)}")
    
    # 2. Deslocamento do Eixo Cartesiano (Centralização nos Baricentros)
    X_mean = X.mean(dim=0, keepdim=True)
    Y_mean = Y.mean(dim=0, keepdim=True)
    
    X_c = X - X_mean
    Y_c = Y - Y_mean
    
    # 3. Operação de Álgebra Linear Fechada (Solução SVD)
    print("[AÇÃO] Executando Singular Value Decomposition (SVD) sobre o cruzamento matricial...")
    M = torch.matmul(X_c.T, Y_c)
    U, S, Vh = torch.linalg.svd(M)
    W = torch.matmul(U, Vh)
    
    # 4. Projeção Vetorial Simulada (Verificação Fechada de Treino)
    Y_pred_c = torch.matmul(X_c, W)
    Y_pred = Y_pred_c + Y_mean
    
    # 5. Avaliação Direcional através do Filtro Topológico CSLS
    print("[AÇÃO] Extraindo correspondências semânticas e validando Hubs...")
    csls_mat = compute_csls(Y_pred, Y, k=5)
    
    best_scores, best_indices = torch.max(csls_mat, dim=1)
    
    acertos = 0
    print("\n--- TOP-20 MAPEAMENTO ESTRITO (CLOSED-SET) ---")
    print(f"{'Origem: Nheengatu':<20} | {'Destino: Português':<20} | {'Fator CSLS':<10} | {'Auditoria'}")
    print("-" * 75)
    
    for i in range(len(vocab_yrl)):
        src_word = vocab_yrl[i]
        pred_word = vocab_pt[best_indices[i].item()]
        true_word = vocab_pt[i]
        score = best_scores[i].item()
        
        is_correct = (pred_word == true_word)
        if is_correct: 
            acertos += 1
            
        if i < 20: 
            status = "Alinhado" if is_correct else f"Erro de Fixação (Padrão: {true_word})"
            print(f"{src_word:<20} | {pred_word:<20} | {score:.4f} | {status}")
            
    print("-" * 75)
    print(f"[DESEMPENHO] Overfitting Estrutural (Base Âncora): {(acertos/len(vocab_yrl))*100:.2f}% ({acertos}/{len(vocab_yrl)})")
    
    # 6. Gravação do peso final da álgebra
    torch.save(W, W_OUT_PATH)
    print(f"\n[SUCESSO] Operação finalizada. Artefato matemático (W) serializado em: {W_OUT_PATH}")

if __name__ == "__main__":
    execute_strict_procrustes()