import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import warnings

# Silenciar avisos e logs de fontes do Matplotlib ou Seaborn
warnings.filterwarnings("ignore")

# Caminhos Dinâmicos Absolutos
PROJECT_ROOT = os.getcwd()
MATRIZ_X_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "matriz_X_yrl.pt")
MATRIZ_Y_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "matriz_Y_pt.pt")
W_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "matriz_W_estrita.pt")

# Caminho da Saída Visual
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
OUT_FILE = os.path.join(OUT_DIR, "pca_alignment.png")

def gerar_grafico_pca():
    print("📈 Iniciando a extração da Prova Visual de Alinhamento (PCA)...")

    # 1. Validação de Arquivos
    if not all(os.path.exists(p) for p in [MATRIZ_X_PATH, MATRIZ_Y_PATH, W_PATH]):
        print("[ERRO FATAL] Tensores ou Matriz W não encontrados. Execute o orthogonal_procrustes.py primeiro.")
        return

    # 2. Carregamento dos Tensores (768 dimensões)
    print("   -> Carregando matrizes do hiperespaço...")
    X = torch.load(MATRIZ_X_PATH)
    Y = torch.load(MATRIZ_Y_PATH)
    W = torch.load(W_PATH)

    # 3. Alinhamento Estrito (Projeção do Nheengatu no Espaço do PT)
    print("   -> Aplicando a translação e rotação da Matriz W...")
    X_mean = X.mean(dim=0, keepdim=True)
    Y_mean = Y.mean(dim=0, keepdim=True)
    
    Y_pred = torch.matmul(X - X_mean, W) + Y_mean

    # Convertendo de Tensores PyTorch para Arrays NumPy
    Y_np = Y.numpy()
    Y_pred_np = Y_pred.numpy()

    # 4. Redução Dimensional Parametrizada
    print("   -> Calculando as Componentes Principais (PCA) conjuntas...")
    # Concatenar para garantir o mesmo sistema de coordenadas na redução
    matriz_combinada = np.vstack([Y_np, Y_pred_np])
    
    pca = PCA(n_components=2)
    matriz_2d = pca.fit_transform(matriz_combinada)
    
    # Separando de volta após a compressão 2D
    tamanho_pt = len(Y_np)
    Y_pca = matriz_2d[:tamanho_pt]
    Y_pred_pca = matriz_2d[tamanho_pt:]

    # 5. Geração do Gráfico de Dispersão
    print("   -> Plotando o gráfico de dispersão...")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 8))
    
    # Plotando Português (Vermelho) e Nheengatu Projetado (Azul)
    sns.scatterplot(
        x=Y_pca[:, 0], y=Y_pca[:, 1], 
        color="#e74c3c", label="Português (Base)", 
        alpha=0.6, s=60, edgecolor="k"
    )
    
    sns.scatterplot(
        x=Y_pred_pca[:, 0], y=Y_pred_pca[:, 1], 
        color="#3498db", label="Nheengatu (Projetado)", 
        alpha=0.6, s=60, edgecolor="k"
    )

    # Estilização para ambiente acadêmico
    plt.title("Isomorfismo Geométrico: Nheengatu Projetado vs. Português", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Componente Principal 1", fontsize=11)
    plt.ylabel("Componente Principal 2", fontsize=11)
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()

    # 6. Exportação em Alta Resolução
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(OUT_FILE, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Prova visual gerada com sucesso e exportada para:\n -> {OUT_FILE}")

if __name__ == "__main__":
    gerar_grafico_pca()