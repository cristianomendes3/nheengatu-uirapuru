import os
import sys
import time
import json
import torch
import torch.nn.functional as F
import warnings

# Supressão de verbosidades de diagnóstico da API do Hugging Face
warnings.filterwarnings("ignore")

# --- RESOLUÇÃO ESTÁTICA DE CAMINHOS ---
PROJECT_ROOT = os.getcwd()
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from models.context_extractor import load_ai_ecosystem, extract_contextual_vector
from alignment.csls_filter import compute_csls

class SemanticExplorerEngine:
    """
    Motor Central de Inferência Semântica Bilíngue.
    
    Esta classe orquestra a fusão entre os modelos Transformers (Hugging Face) e o 
    espaço vetorial latente alinhado via Procrustes Ortogonal. Ela gerencia a 
    memória RAM (cache de tensores), a dedução em tempo real de palavras 
    desconhecidas (Zero-Shot) e o cálculo de vizinhança topológica (CSLS).
    """
    
    def __init__(self):
        """
        Construtor do Motor de Inferência.
        
        Carrega as Redes Neurais, as Matrizes de Ancoragem (X, Y) e a Matriz de 
        Inteligência (W). Realiza a fusão dos vocabulários (Base + Piscina In-Domain)
        e aplica a Deduplicação Independente, garantindo que o espaço vetorial 
        não possua sobreposições matemáticas viciadas.
        """
        print("[Engine] Iniciando a ignição do Motor de Inferência Semântica...")
        start_load = time.time()
        
        self.tok_yrl, self.mod_yrl, self.tok_pt, self.mod_pt = load_ai_ecosystem()
        
        # Desserialização dos Artefatos Geométricos e da Matriz de Rotação (W)
        X_anc = torch.load(os.path.join(PROJECT_ROOT, "data", "processed", "matriz_X_yrl.pt"))
        Y_anc = torch.load(os.path.join(PROJECT_ROOT, "data", "processed", "matriz_Y_pt.pt"))
        self.W = torch.load(os.path.join(PROJECT_ROOT, "data", "processed", "matriz_W_estrita.pt"))
        
        with open(os.path.join(PROJECT_ROOT, "data", "processed", "embeddings_extraidos.json"), 'r', encoding='utf-8') as f:
            vocab_anc = json.load(f)
            
        X_pool = torch.load(os.path.join(PROJECT_ROOT, "data", "processed", "pool_X_yrl.pt"))
        Y_pool = torch.load(os.path.join(PROJECT_ROOT, "data", "processed", "pool_Y_pt.pt"))
        
        with open(os.path.join(PROJECT_ROOT, "data", "processed", "pool_vocab.json"), 'r', encoding='utf-8') as f:
            vocab_pool = json.load(f)

        # Extração Bruta e Unificação de Léxicos
        vocab_yrl_bruto = [item.get("alvo_yrl", item.get("yrl", "")) for item in vocab_anc] + \
                          [item.get("yrl", "") for item in vocab_pool]
        vocab_pt_bruto = [item.get("alvo_pt", item.get("pt", "")) for item in vocab_anc] + \
                         [item.get("pt", "") for item in vocab_pool]
                         
        X_bruto = torch.cat([X_anc, X_pool], dim=0)
        Y_bruto = torch.cat([Y_anc, Y_pool], dim=0)
        
        # Desacoplamento Matemático: Garante a bi-jeção mapeando índices únicos
        self.vocab_yrl, indices_yrl, vistos_yrl = [], [], set()
        for i, word in enumerate(vocab_yrl_bruto):
            if word not in vistos_yrl:
                vistos_yrl.add(word)
                self.vocab_yrl.append(word)
                indices_yrl.append(i)
        self.X_global = X_bruto[indices_yrl]

        self.vocab_pt, indices_pt, vistos_pt = [], [], set()
        for i, word in enumerate(vocab_pt_bruto):
            if word not in vistos_pt:
                vistos_pt.add(word)
                self.vocab_pt.append(word)
                indices_pt.append(i)
        self.Y_global = Y_bruto[indices_pt]

        # Fixação dos Centros de Gravidade baseados unicamente no conjunto âncora
        self.X_mean = X_anc.mean(dim=0, keepdim=True)
        self.Y_mean = Y_anc.mean(dim=0, keepdim=True)
            
        print(f"[Engine] Ignição concluída em {time.time() - start_load:.2f}s. Nheengatu: {len(self.vocab_yrl)} | Português: {len(self.vocab_pt)}")

    def _get_vector(self, query: str, lang: str):
        """
        Recupera ou infere as coordenadas hiperdimensionais de uma palavra.
        
        Se a palavra consta na memória (vocab_yrl/vocab_pt), retorna o vetor pré-computado.
        Caso contrário, aciona a IA correspondente para realizar a extração contextual Zero-Shot.
        """
        query = query.strip().lower()
        if lang == "yrl" and query in self.vocab_yrl:
            return self.X_global[self.vocab_yrl.index(query)].unsqueeze(0)
        if lang == "pt" and query in self.vocab_pt:
            return self.Y_global[self.vocab_pt.index(query)].unsqueeze(0)
            
        context = f"Aé unheengatu {query}." if lang == "yrl" else f"A palavra em português é {query}."
        vec = extract_contextual_vector(context, query, self.tok_yrl, self.mod_yrl) if lang == "yrl" else \
              extract_contextual_vector(context, query, self.tok_pt, self.mod_pt)
            
        if vec is None:
            raise ValueError(f"Falha ao extrair vetor latente para a representação: {query}")
        return vec.unsqueeze(0)

    def search(self, query: str, direction: str = "yrl_to_pt", top_k: int = 5):
        """
        Executa a busca topológica e o pareamento semântico no espaço vetorial alinhado.
        
        Centraliza o vetor de entrada, aplica a rotação ortogonal estrita (W ou W.T) e 
        calcula os vizinhos mais próximos no hiperespaço alvo utilizando similaridade de 
        cosseno pura e o filtro corretor CSLS.
        
        Args:
            query (str): A palavra ou termo de busca.
            direction (str): Fluxo de tradução ('yrl_to_pt' ou 'pt_to_yrl').
            top_k (int): Número de vizinhos hiperdimensionais a serem retornados.
            
        Returns:
            dict: Relatório estruturado contendo as métricas XAI e os tensores 
                  brutos para renderização no Frontend (Dashboard).
        """
        start_inference = time.time()
        query = query.strip().lower()
        
        # Configuração do fluxo bidirecional e definição da rotação (W e W Transposta)
        if direction == "yrl_to_pt":
            input_lang, source_mean, target_mean = "yrl", self.X_mean, self.Y_mean
            rotation_matrix, target_matrix, target_vocab = self.W, self.Y_global, self.vocab_pt
        elif direction == "pt_to_yrl":
            input_lang, source_mean, target_mean = "pt", self.Y_mean, self.X_mean
            rotation_matrix, target_matrix, target_vocab = self.W.T, self.X_global, self.vocab_yrl
        else:
            raise ValueError("Direção de alinhamento geométrica inválida.")

        # 1. Projeção Ortogonal Dinâmica (O salto entre os espaços latentes)
        input_tensor = self._get_vector(query, input_lang)
        tensor_centralizado = input_tensor - source_mean
        tensor_rotacionado = torch.matmul(tensor_centralizado, rotation_matrix)
        tensor_projetado = tensor_rotacionado + target_mean
        
        # 2. Filtragem Topológica (CSLS) e Distância Euclidiana Pura (Cosseno)
        csls_scores = compute_csls(tensor_projetado, target_matrix, k=5)
        best_scores_csls, best_indices = torch.topk(csls_scores[0], k=top_k)
        
        proj_norm = F.normalize(tensor_projetado, p=2, dim=1)
        target_norm = F.normalize(target_matrix, p=2, dim=1)
        cos_sims_matrix = torch.matmul(proj_norm, target_norm.transpose(0, 1))[0]
        
        # 3. Empacotamento de Resultados para Serialização (XAI e Plotly)
        resultados = []
        for score_csls, idx in zip(best_scores_csls.tolist(), best_indices.tolist()):
            resultados.append({
                "palavra": target_vocab[idx],
                "score_cosseno": round(cos_sims_matrix[idx].item(), 4),
                "score_csls": round(score_csls, 4),
                "vetor_raw": target_matrix[idx].tolist()  # Tensor de 768d exposto para o PCA Frontend
            })
            
        tempo_total = time.time() - start_inference
        
        return {
            "query_original": query,
            "direcao": direction,
            "tempo_inferencia_ms": round(tempo_total * 1000, 2),
            "top_k_resultados": resultados,
            "metadados": {
                "tensor_query": tensor_projetado.squeeze(0).tolist()
            }
        }