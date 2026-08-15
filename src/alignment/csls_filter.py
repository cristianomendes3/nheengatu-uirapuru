import torch
import torch.nn.functional as F

def compute_csls(source_embs: torch.Tensor, target_embs: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    Calcula a matriz de similaridade CSLS (Cross-Domain Similarity Local Scaling) 
    entre dois conjuntos de embeddings alinhados.
    
    O CSLS mitiga o problema dos 'hubs' semânticos (palavras que aparecem como 
    tradução para muitas outras devido à densidade geométrica do espaço vetorial).
    
    Argumentos:
        source_embs (torch.Tensor): Tensores de origem (Nheengatu rotacionado) de shape (N, D).
        target_embs (torch.Tensor): Tensores de destino (Português) de shape (M, D).
        k (int): Número de vizinhos mais próximos para considerar na penalidade (padrão=5).
        
    Retorna:
        torch.Tensor: Matriz de similaridade CSLS de shape (N, M).
    """
    
    # 1. Normalização L2 dos vetores (garante que a similaridade será de cosseno)
    source_norm = F.normalize(source_embs, p=2, dim=1)
    target_norm = F.normalize(target_embs, p=2, dim=1)
    
    # 2. Calcula a matriz de similaridade de cosseno absoluta (N, M)
    sim_matrix = torch.mm(source_norm, target_norm.t())
    
    # Proteção para bases pequenas
    n_source, n_target = sim_matrix.shape
    k_target = min(k, n_target)
    k_source = min(k, n_source)
    
    # 3. Penalidade da vizinhança do Destino
    topk_sim_target = torch.topk(sim_matrix, k=k_target, dim=1).values
    r_source = topk_sim_target.mean(dim=1, keepdim=True)  # Shape (N, 1)
    
    # 4. Penalidade da vizinhança de Origem
    topk_sim_source = torch.topk(sim_matrix, k=k_source, dim=0).values
    r_target = topk_sim_source.mean(dim=0, keepdim=True)  # Shape (1, M)
    
    # 5. Aplica a fórmula do CSLS
    csls_matrix = 2 * sim_matrix - r_source - r_target
    
    return csls_matrix

# Bloco protegido: só executa se o script for chamado diretamente (teste isolado)
if __name__ == "__main__":
    print("Iniciando teste de estresse e calibração de K...")
    N, M, D = 100, 120, 300
    
    torch.manual_seed(42)
    dummy_source = torch.randn(N, D)
    dummy_target = torch.randn(M, D)
    
    k_values = [1, 5, 10]
    
    for k_val in k_values:
        csls_result = compute_csls(dummy_source, dummy_target, k=k_val)
        print(f"\nTeste com K={k_val}:")
        print(f"Shape: {csls_result.shape} | Max: {csls_result.max().item():.4f} | Min: {csls_result.min().item():.4f}")