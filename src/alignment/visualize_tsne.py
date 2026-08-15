import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import warnings

# Silenciar avisos e logs de fontes do Matplotlib, Seaborn ou scikit-learn
warnings.filterwarnings("ignore")

# Caminhos Dinâmicos Absolutos
PROJECT_ROOT = os.getcwd()
MATRIZ_X_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "matriz_X_yrl.pt")
MATRIZ_Y_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "matriz_Y_pt.pt")
W_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "matriz_W_estrita.pt")
VOCAB_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "embeddings_extraidos.json")

# Caminho da Saída Visual
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
OUT_FILE = os.path.join(OUT_DIR, "tsne_semantic_clusters.png")

def gerar_grafico_tsne():
    print("🧩 Iniciando a extração do Mapa Topológico Local (t-SNE)...")

    # 1. Validação de Arquivos
    if not all(os.path.exists(p) for p in [MATRIZ_X_PATH, MATRIZ_Y_PATH, W_PATH, VOCAB_PATH]):
        print("[ERRO FATAL] Tensores, Matriz W ou Vocabulário ausentes. Verifique a pasta data/processed/.")
        return

    # 2. Carregamento dos Artefatos
    print("   -> Carregando matrizes de 768 dimensões e léxico Nheengatu...")
    X = torch.load(MATRIZ_X_PATH)
    Y = torch.load(MATRIZ_Y_PATH)
    W = torch.load(W_PATH)

    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
        # Extrai apenas as palavras em Nheengatu para legendar o gráfico
        palavras_yrl = [item.get("alvo_yrl", "") for item in vocab]

    # 3. Projeção Ortogonal Estrita (O Salto Dimensional)
    print("   -> Calculando a projeção para o espaço alinhado...")
    X_mean = X.mean(dim=0, keepdim=True)
    Y_mean = Y.mean(dim=0, keepdim=True)
    
    Y_pred = torch.matmul(X - X_mean, W) + Y_mean
    Y_pred_np = Y_pred.numpy()

    # 4. Redução Não-Linear (t-SNE)
    print("   -> Executando t-SNE (Perplexidade ajustada para base restrita)...")
    # Perplexity=10 é o ideal para nossa base de ~100 pontos.
    tsne = TSNE(n_components=2, perplexity=10, random_state=42, init='pca', learning_rate='auto')
    matriz_2d = tsne.fit_transform(Y_pred_np)

    # 5. Renderização do Gráfico e Anotação Semântica
    print("   -> Desenhando clusters e inserindo as anotações textuais...")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 10))

    # Plotagem dos marcadores (Apenas Nheengatu Projetado)
    sns.scatterplot(
        x=matriz_2d[:, 0], y=matriz_2d[:, 1], 
        color="#3498db", alpha=0.7, s=80, edgecolor="k"
    )

    # Iteração para anotar cada ponto com a sua respectiva palavra em Nheengatu
    for i, palavra in enumerate(palavras_yrl):
        plt.annotate(
            palavra,
            (matriz_2d[i, 0], matriz_2d[i, 1]),
            xytext=(6, 3), # Deslocamento do texto em relação ao ponto para evitar sobreposição
            textcoords='offset points',
            fontsize=9,
            alpha=0.85,
            weight='medium'
        )

    # Estilização do Relatório
    plt.title("Mapa Topológico (t-SNE): Preservação Semântica do Nheengatu Projetado", fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Dimensão t-SNE 1 (Afinidade Local)", fontsize=12)
    plt.ylabel("Dimensão t-SNE 2 (Afinidade Local)", fontsize=12)
    plt.tight_layout()

    # 6. Exportação em Alta Resolução
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(OUT_FILE, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Mapa semântico gerado com sucesso e exportado para:\n -> {OUT_FILE}")

if __name__ == "__main__":
    gerar_grafico_tsne()