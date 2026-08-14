import os
import torch
import torch.nn.functional as F

MATRIZ_W_PATH = "data/processed/matriz_W_rotacao.pt"
X_VAL_PATH = "data/processed/X_val.pt"
Y_VAL_PATH = "data/processed/Y_val.pt"
CENTROIDES_PATH = "data/processed/centroides.pt"

def executar_projecao_cruzada():
    print("[INFO] Iniciando o Pipeline de Projeção Cruzada e Avaliação (Zero-Shot)...")
    
    # 1. Carregamento de Contenção
    paths = [MATRIZ_W_PATH, X_VAL_PATH, Y_VAL_PATH, CENTROIDES_PATH]
    if not all(os.path.exists(p) for p in paths):
        print("[ERRO] Artefatos de validação ou matriz de rotação não encontrados. Execute procrustes.py primeiro.")
        return

    W = torch.load(MATRIZ_W_PATH)
    X_val = torch.load(X_VAL_PATH)
    Y_val = torch.load(Y_VAL_PATH)
    centroides = torch.load(CENTROIDES_PATH)
    
    X_mean = centroides["X_mean"]
    Y_mean = centroides["Y_mean"]
    
    print(f" -> Conjunto de Validação Carregado: {X_val.shape[0]} pares cegos de teste.")
    
    # 2. Centralização Pré-Projeção (Subtração do centroide de treino)
    X_centered_val = X_val - X_mean
    
    # 3. Projeção Linear (O Salto Hiperdimensional: Y_hat_centered = X_val_centered @ W)
    print("[INFO] Aplicando a Matriz de Rotação sobre o Nheengatu de Validação...")
    Y_hat_centered = torch.matmul(X_centered_val, W)
    
    # 4. Reversão de Escala / Descentralização (Adição do centroide do Português)
    Y_hat = Y_hat_centered + Y_mean
    
    # 5. Métrica de Similaridade de Cosseno (Par a par)
    # torch.nn.functional.cosine_similarity calcula a distância angular entre os vetores preditos e os reais
    similares = F.cosine_similarity(Y_hat, Y_val, dim=1)
    
    # 6. Geração de Relatório de Diagnóstico
    media_cosseno = similares.mean().item()
    min_cosseno = similares.min().item()
    max_cosseno = similares.max().item()
    
    print("\n--- RELATÓRIO DE DIAGNÓSTICO (ZERO-SHOT) ---")
    print(f"Similaridade de Cosseno Média (Validação): {media_cosseno:.4f}")
    print(f"Similaridade Mínima: {min_cosseno:.4f} | Máxima: {max_cosseno:.4f}")
    
    print("\n--- AMOSTRAGEM INDIVIDUAL DE PARES PROJETADOS ---")
    for i in range(X_val.shape[0]):
        print(f" Par [{i+1}] -> Similaridade de Cosseno: {similares[i].item():.4f}")

    print(f"\n[SUCESSO] Avaliação de projeção cruzada concluída com êxito.")

if __name__ == "__main__":
    executar_projecao_cruzada()