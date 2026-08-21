import torch
import torch.nn.functional as F

def compute_csls(source_embs: torch.Tensor, target_embs: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    Calcula a matriz de similaridade CSLS (Cross-Domain Similarity Local Scaling) 
    entre dois conjuntos de embeddings alinhados no espaço euclidiano.
    
    O CSLS atua como um filtro topológico. Ele mitiga ativamente o fenômeno do 
    'Hubness' — onde palavras específicas tornam-se ímãs hiperdimensionais devido 
    à assimetria de densidade geométrica —, penalizando pares que possuem vizinhanças 
    compostas por distâncias angulares artificialmente estreitas.
    
    Args:
        source_embs (torch.Tensor): Matriz de tensores projetados (Origem) de shape (N, D).
        target_embs (torch.Tensor): Matriz de tensores ancorados (Destino) de shape (M, D).
        k (int): K-vizinhos mais próximos para mensuração da densidade (padrão = 5).
        
    Returns:
        torch.Tensor: Matriz penalizada e normalizada de similaridade bidirecional CSLS (N, M).
    """
    
    # 1. Normalização L2 para forçar o limite geométrico a uma esfera unitária
    source_norm = F.normalize(source_embs, p=2, dim=1)
    target_norm = F.normalize(target_embs, p=2, dim=1)
    
    # 2. Produto escalar entre esferas normalizadas resulta na Similaridade de Cosseno (N, M)
    sim_matrix = torch.mm(source_norm, target_norm.t())
    
    # Prevenção de limite de busca (K) para vocabulários em fase de testes (Low-Resource)
    n_source, n_target = sim_matrix.shape
    k_target = min(k, n_target)
    k_source = min(k, n_source)
    
    # 3. Penalidade r_T(y): Quão densa é a vizinhança na língua alvo?
    topk_sim_target = torch.topk(sim_matrix, k=k_target, dim=1).values
    r_source = topk_sim_target.mean(dim=1, keepdim=True)  # Shape (N, 1)
    
    # 4. Penalidade r_S(x): Quão densa é a vizinhança na língua origem?
    topk_sim_source = torch.topk(sim_matrix, k=k_source, dim=0).values
    r_target = topk_sim_source.mean(dim=0, keepdim=True)  # Shape (1, M)
    
    # 5. Aplicação da função simétrica CSLS
    csls_matrix = 2 * sim_matrix - r_source - r_target
    
    return csls_matrix

if __name__ == "__main__":
    print("[INFO] Iniciando rotina de estresse e calibração de limite K...")
    N, M, D = 100, 120, 300
    
    torch.manual_seed(42)
    dummy_source = torch.randn(N, D)
    dummy_target = torch.randn(M, D)
    
    k_values = [1, 5, 10]
    
    for k_val in k_values:
        csls_result = compute_csls(dummy_source, dummy_target, k=k_val)
        print(f"\nTeste Paramétrico [K={k_val}]:")
        print(f"Formato da Matriz: {csls_result.shape} | Valor Máx: {csls_result.max().item():.4f} | Valor Mín: {csls_result.min().item():.4f}")