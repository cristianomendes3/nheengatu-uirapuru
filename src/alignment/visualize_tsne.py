import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import warnings

# Opressão de ruídos de saída e loggings do pipeline de DataViz
warnings.filterwarnings("ignore")

# Orientação referencial global e roteamento de arquivos físicos
PROJECT_ROOT = os.getcwd()
MATRIZ_X_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "matriz_X_yrl.pt")
MATRIZ_Y_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "matriz_Y_pt.pt")
W_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "matriz_W_estrita.pt")
VOCAB_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "embeddings_extraidos.json")

# Estruturação de rota estática para a Dashboard
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
OUT_FILE = os.path.join(OUT_DIR, "tsne_semantic_clusters.png")

def gerar_grafico_tsne():
    """
    Renderiza um Mapeamento Cartográfico Global (Topológico) através de algoritmos
    não lineares.

    Diferente do PCA, a t-Distributed Stochastic Neighbor Embedding (t-SNE) é 
    responsável por enfatizar densidades e afinidades locais, organizando os dados de
    768 dimensões em "clusters semânticos" isolados visíveis (Ex: Animais, Verbos, etc).
    Atua como base de justificativa topológica e Explicabilidade de IA (XAI).

    Returns:
        None: Grava um arquivo fotográfico formatado na pasta de Outputs.
    """
    print("[INFO] Preparando processo iterativo e topológico local (t-SNE)...")

    # Mecanismo de defesa contra execuções fora de ordem
    if not all(os.path.exists(p) for p in [MATRIZ_X_PATH, MATRIZ_Y_PATH, W_PATH, VOCAB_PATH]):
        print("[ERRO FATAL] Tensores paramétricos, matrizes de cálculo ou chaves lexicais ausentes no subdiretório de processamento.")
        return

    # Injeção e formatação cruzada do Dicionário vs Matrizes
    print("[AÇÃO] Indexando dados dimensionais e metadados descritivos...")
    X = torch.load(MATRIZ_X_PATH)
    Y = torch.load(MATRIZ_Y_PATH)
    W = torch.load(W_PATH)

    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
        palavras_yrl = [item.get("alvo_yrl", "") for item in vocab]

    # Modelagem algébrica via rotação W para posicionamento
    print("[AÇÃO] Forçando salto topológico e alinhamento dimensional...")
    X_mean = X.mean(dim=0, keepdim=True)
    Y_mean = Y.mean(dim=0, keepdim=True)
    
    Y_pred = torch.matmul(X - X_mean, W) + Y_mean
    Y_pred_np = Y_pred.numpy()

    # Conversão de Coordenadas Incompressíveis e Configuração de Redução
    print("[AÇÃO] Compilando Manifold T-Distributed... (Parâmetros balanceados para escopo reduzido)")
    # Uso de Perplexity=10 é calibrado e obrigatório devido ao tamanho diminuto (Low-Resource) da matriz original.
    tsne = TSNE(n_components=2, perplexity=10, random_state=42, init='pca', learning_rate='auto')
    matriz_2d = tsne.fit_transform(Y_pred_np)

    # Pipeline de Rasterização do Cartograma
    print("[AÇÃO] Gerando polígonos de agrupamento (Clusterização Semântica)...")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 10))

    sns.scatterplot(
        x=matriz_2d[:, 0], y=matriz_2d[:, 1], 
        color="#3498db", alpha=0.7, s=80, edgecolor="k"
    )

    # Disposição Textual Dinâmica (Evita esmagamento de coordenadas)
    for i, palavra in enumerate(palavras_yrl):
        plt.annotate(
            palavra,
            (matriz_2d[i, 0], matriz_2d[i, 1]),
            xytext=(6, 3), 
            textcoords='offset points',
            fontsize=9,
            alpha=0.85,
            weight='medium'
        )

    # Arquitetura Informativa da Plotagem
    plt.title("Preservação Semântica e Clusterização Local do Idioma Nheengatu (t-SNE)", fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Métrica de Afinidade Primária", fontsize=12)
    plt.ylabel("Métrica de Afinidade Secundária", fontsize=12)
    plt.tight_layout()

    # Gravação e encerramento
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(OUT_FILE, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SUCESSO] Renderização Topológica de Clusterização gerada com sucesso em: {OUT_FILE}")

if __name__ == "__main__":
    gerar_grafico_tsne()