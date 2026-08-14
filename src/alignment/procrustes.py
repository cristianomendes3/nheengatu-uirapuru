import os
import torch

MATRIZ_X_PATH = "data/processed/matriz_X_yrl.pt"
MATRIZ_Y_PATH = "data/processed/matriz_Y_pt.pt"
MATRIZ_W_OUT = "data/processed/matriz_W_rotacao.pt"

# Novos caminhos para a exportação modular
X_TRAIN_OUT = "data/processed/X_train.pt"
Y_TRAIN_OUT = "data/processed/Y_train.pt"
X_VAL_OUT = "data/processed/X_val.pt"
Y_VAL_OUT = "data/processed/Y_val.pt"
CENTROIDES_OUT = "data/processed/centroides.pt"

def calcular_matriz_procrustes_regularizada():
    print("[INFO] Iniciando o Motor Matemático: Alinhamento de Procrustes Ortogonal Regularizado")
    
    if not os.path.exists(MATRIZ_X_PATH) or not os.path.exists(MATRIZ_Y_PATH):
        print("[ERRO] Matrizes de ancoragem não encontradas.")
        return

    X = torch.load(MATRIZ_X_PATH)
    Y = torch.load(MATRIZ_Y_PATH)
    
    # 1. Isolamento de Validação (Split 80/20)
    # A semente manual garante reprodutibilidade rigorosa no ambiente de pesquisa.
    torch.manual_seed(42) 
    indices = torch.randperm(X.shape[0])
    split_idx = int(X.shape[0] * 0.8)
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    X_train, Y_train = X[train_indices], Y[train_indices]
    X_val, Y_val = X[val_indices], Y[val_indices]
    
    # 2. Cálculo dos Centroides (Médias)
    X_mean = X_train.mean(dim=0, keepdim=True)
    Y_mean = Y_train.mean(dim=0, keepdim=True)
    
    # 3. Centralização Vetorial (Mean Subtraction)
    X_centered = X_train - X_mean
    Y_centered = Y_train - Y_mean
    
    # 4. SVD Regularizado
    M = torch.matmul(Y_centered.T, X_centered)
    U, S, Vh = torch.linalg.svd(M)
    
    # 5. Atualização da Matriz de Rotação (W)
    W = torch.matmul(U, Vh)
    
    print("\n--- AUDITORIA GEOMÉTRICA DO OPERADOR ---")
    print(f"Treino (SVD): {X_train.shape[0]} sentenças | Validação (Cega): {X_val.shape[0]} sentenças")
    # A soma das médias da matriz centralizada deve tender a zero.
    print(f"Centroide X Regularizado (Média Global): {X_centered.mean().item():.8f} (Esperado ~0.0)")
    print(f"Formato da Matriz de Rotação (W): {list(W.shape)}")
    
    assert W.shape == (768, 768), "[FATAL] A matriz de rotação não possui dimensionalidade 768x768."
    
    # 6. Exportação Modular
    torch.save(W, MATRIZ_W_OUT)
    torch.save(X_train, X_TRAIN_OUT)
    torch.save(Y_train, Y_TRAIN_OUT)
    torch.save(X_val, X_VAL_OUT)
    torch.save(Y_val, Y_VAL_OUT)
    # Salva os centroides para que os testes das próximas semanas possam projetar palavras inéditas
    torch.save({"X_mean": X_mean, "Y_mean": Y_mean}, CENTROIDES_OUT)
    
    print(f"\n[SUCESSO] O núcleo matemático regularizado foi forjado e salvo.")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(MATRIZ_W_OUT), exist_ok=True)
    calcular_matriz_procrustes_regularizada()