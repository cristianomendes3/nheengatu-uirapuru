import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import warnings

# Bloqueio de logs oriundos do Matplotlib/Seaborn durante plotting
warnings.filterwarnings("ignore")

# Identificação e vinculação de caminhos locais
PROJECT_ROOT = os.getcwd()
MATRIZ_X_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "matriz_X_yrl.pt")
MATRIZ_Y_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "matriz_Y_pt.pt")
W_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "matriz_W_estrita.pt")

# Apontamentos de gravação de imagens
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
OUT_FILE = os.path.join(OUT_DIR, "pca_alignment.png")

def gerar_grafico_pca():
    """
    Renderiza uma representação Euclidiana comprimida do alinhamento geométrico.

    Utiliza Análise de Componentes Principais (PCA) para reduzir os vetores de
    768 dimensões para planos visuais de 2 componentes. Avalia a eficácia da matriz W 
    em transladar e comprimir os tensores do Nheengatu de forma a espelhar a distribuição
    dos tensores nativos da língua Portuguesa.

    Returns:
        None: Exporta a renderização analítica final no formato .png em alta resolução.
    """
    print("[INFO] Acionando rotina de documentação visual matricial (PCA-2D)...")

    # Validação rigorosa do pipeline anterior
    if not all(os.path.exists(p) for p in [MATRIZ_X_PATH, MATRIZ_Y_PATH, W_PATH]):
        print("[ERRO FATAL] O artefato W ou as tensores de entrada não estão disponíveis. O script 'orthogonal_procrustes.py' deve ser executado primariamente.")
        return

    # Descompactação das coordenadas 
    print("[AÇÃO] Carregamento matricial para a memória RAM (768 Dimensões)...")
    X = torch.load(MATRIZ_X_PATH)
    Y = torch.load(MATRIZ_Y_PATH)
    W = torch.load(W_PATH)

    # Execução do Salto Geométrico Parametrizado
    print("[AÇÃO] Rotacionando o léxico Nheengatu no interior do espaço referencial...")
    X_mean = X.mean(dim=0, keepdim=True)
    Y_mean = Y.mean(dim=0, keepdim=True)
    
    Y_pred = torch.matmul(X - X_mean, W) + Y_mean

    # Modulação de matrizes de Pytorch para o framework NumPy
    Y_np = Y.numpy()
    Y_pred_np = Y_pred.numpy()

    # Operação Estatística Híbrida
    print("[AÇÃO] Compactando componentes lineares unificadas (PCA Conjunto)...")
    matriz_combinada = np.vstack([Y_np, Y_pred_np])
    
    pca = PCA(n_components=2)
    matriz_2d = pca.fit_transform(matriz_combinada)
    
    # Desmembramento espacial das matrizes para exibição individual
    tamanho_pt = len(Y_np)
    Y_pca = matriz_2d[:tamanho_pt]
    Y_pred_pca = matriz_2d[tamanho_pt:]

    # Plotagem utilizando o Seaborn Framework
    print("[AÇÃO] Renderizando imagem de dispersão relacional...")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 8))
    
    sns.scatterplot(
        x=Y_pca[:, 0], y=Y_pca[:, 1], 
        color="#e74c3c", label="Português (Referência Fixa)", 
        alpha=0.6, s=60, edgecolor="k"
    )
    
    sns.scatterplot(
        x=Y_pred_pca[:, 0], y=Y_pred_pca[:, 1], 
        color="#3498db", label="Nheengatu (Projeção Alinhada)", 
        alpha=0.6, s=60, edgecolor="k"
    )

    # Hierarquia e titulação formal da exportação
    plt.title("Validação de Isomorfismo: Reflexão Vetorial via Matriz Procrustes", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Vetor de Redução 1 (PCA)", fontsize=11)
    plt.ylabel("Vetor de Redução 2 (PCA)", fontsize=11)
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()

    # Serialização estática
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(OUT_FILE, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SUCESSO] Impressão final do PCA gerada na rota: {OUT_FILE}")

if __name__ == "__main__":
    gerar_grafico_pca()