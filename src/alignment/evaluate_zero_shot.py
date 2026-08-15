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

# Matrizes Âncora (Usadas apenas para descobrir o "Centro de Gravidade" do Treino)
MATRIZ_X_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "matriz_X_yrl.pt")
MATRIZ_Y_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "matriz_Y_pt.pt")

# A Matriz de Inteligência (A Rotação Aprendida)
W_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "matriz_W_estrita.pt")

# Os Dados Inéditos de Teste (A Piscina)
POOL_X_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "pool_X_yrl.pt")
POOL_Y_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "pool_Y_pt.pt")
VOCAB_POOL_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "pool_vocab.json")

def evaluate_zero_shot():
    print("🚀 Iniciando Teste de Fogo: Generalização Zero-Shot...")
    
    # 1. Carregamento dos Artefatos
    X_anc = torch.load(MATRIZ_X_PATH)
    Y_anc = torch.load(MATRIZ_Y_PATH)
    W = torch.load(W_PATH)
    
    X_test = torch.load(POOL_X_PATH)
    Y_test = torch.load(POOL_Y_PATH)
    
    with open(VOCAB_POOL_PATH, 'r', encoding='utf-8') as f:
        vocab_test = json.load(f)
        
    print(f"📚 Dados Inéditos Carregados: {X_test.shape[0]} palavras.")
    
    # 2. Centralização Baseada no Treino (Crucial para a Geometria)
    X_train_mean = X_anc.mean(dim=0, keepdim=True)
    Y_train_mean = Y_anc.mean(dim=0, keepdim=True)
    
    X_test_c = X_test - X_train_mean
    
    # 3. O Salto Hiperdimensional (Projeção)
    Y_pred_c = torch.matmul(X_test_c, W)
    Y_pred = Y_pred_c + Y_train_mean
    
    # 4. Busca da Tradução (Usando o filtro CSLS para penalizar Hubs)
    print("🔍 Procurando traduções mais próximas no hiperespaço...")
    csls_mat = compute_csls(Y_pred, Y_test, k=5)
    
    # Avaliando Métricas (Top-1, Top-3, Top-5)
    top1_acertos = 0
    top3_acertos = 0
    top5_acertos = 0
    
    total = len(vocab_test)
    
    print("\n🕵️ --- AMOSTRAGEM DE TRADUÇÕES INÉDITAS --- 🕵️")
    print(f"{'Nheengatu (Input)':<20} | {'Previsão da IA':<20} | {'Gabarito (Real)':<20} | {'Status'}")
    print("-" * 80)
    
    for i in range(total):
        # Pega as 5 melhores traduções para a palavra 'i'
        scores, indices = torch.topk(csls_mat[i], k=5)
        indices = indices.tolist()
        
        src_word = vocab_test[i]['yrl']
        true_word = vocab_test[i]['pt']
        best_pred_word = vocab_test[indices[0]]['pt']
        
        # Verifica acertos
        if i == indices[0]: top1_acertos += 1
        if i in indices[:3]: top3_acertos += 1
        if i in indices[:5]: top5_acertos += 1
            
        # Imprime uma amostra de 15 palavras aleatórias/primeiras
        if i < 15:
            status = "🟢 Exato" if i == indices[0] else ("🟡 Top-5" if i in indices else "🔴 Errou")
            print(f"{src_word:<20} | {best_pred_word:<20} | {true_word:<20} | {status}")
            
    print("-" * 80)
    print("\n📈 --- RESULTADOS GLOBAIS DE GENERALIZAÇÃO --- 📈")
    print(f"Acurácia Top-1 (Acertou em cheio): {(top1_acertos / total) * 100:.2f}% ({top1_acertos}/{total})")
    print(f"Acurácia Top-3 (Estava no pódio) : {(top3_acertos / total) * 100:.2f}% ({top3_acertos}/{total})")
    print(f"Acurácia Top-5 (Estava no radar) : {(top5_acertos / total) * 100:.2f}% ({top5_acertos}/{total})")
    
    if (top1_acertos / total) > 0.10:
        print("\n🎉 Veredito: SUCESSO! A matriz aprendeu a mapear as línguas!")
    else:
        print("\n⚠️ Veredito: A generalização foi baixa. O modelo sofre de Overfitting (precisaria de uma base âncora maior que 97 palavras).")

if __name__ == "__main__":
    evaluate_zero_shot()