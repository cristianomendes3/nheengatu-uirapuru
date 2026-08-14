import json
import torch
import os

ARQUIVO_JSON = "data/processed/embeddings_extraidos.json"
MATRIZ_X_OUT = "data/processed/matriz_X_yrl.pt"
MATRIZ_Y_OUT = "data/processed/matriz_Y_pt.pt"

def construir_matrizes_ancoragem():
    print(f"[INFO] Lendo tensores extraídos de {ARQUIVO_JSON}...")
    
    if not os.path.exists(ARQUIVO_JSON):
        print(f"[ERRO] O arquivo {ARQUIVO_JSON} não foi encontrado. Execute extraction_script.py primeiro.")
        return
        
    with open(ARQUIVO_JSON, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    lista_vetores_x = []
    lista_vetores_y = []
    
    for item in dataset:
        lista_vetores_x.append(item['vetor_yrl'])
        lista_vetores_y.append(item['vetor_pt'])
        
    # Conversão de listas nativas do Python para Tensores do PyTorch
    matriz_X = torch.tensor(lista_vetores_x, dtype=torch.float32)
    matriz_Y = torch.tensor(lista_vetores_y, dtype=torch.float32)
    
    print("\n--- AUDITORIA GEOMÉTRICA ---")
    print(f"Matriz X (Nheengatu) : Formato {list(matriz_X.shape)} | Dimensões: {matriz_X.shape[1]}")
    print(f"Matriz Y (Português) : Formato {list(matriz_Y.shape)} | Dimensões: {matriz_Y.shape[1]}")
    
    # Validação Estrutural Rigorosa
    assert matriz_X.shape == matriz_Y.shape, "[FATAL] As matrizes possuem tamanhos diferentes. O alinhamento irá falhar!"
    assert matriz_X.shape[1] == 768, "[FATAL] A dimensionalidade da camada oculta não é 768."
    assert matriz_X.shape[0] == 97, f"[ALERTA] Volume de sentenças divergente. Esperado: 97, Obtido: {matriz_X.shape[0]}"
    
    # Exportação Binária PyTorch
    torch.save(matriz_X, MATRIZ_X_OUT)
    torch.save(matriz_Y, MATRIZ_Y_OUT)
    
    print(f"\n[SUCESSO] Matrizes ancoradas e compiladas com sucesso.")
    print(f"-> Salvo: {MATRIZ_X_OUT}")
    print(f"-> Salvo: {MATRIZ_Y_OUT}")

if __name__ == "__main__":
    construir_matrizes_ancoragem()