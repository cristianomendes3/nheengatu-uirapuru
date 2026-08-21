import torch
import torch.nn.functional as F
from context_extractor import load_ai_ecosystem, extract_contextual_vector

def comparar_impacto_contexto():
    """
    Executa diagnóstico matemático sobre a maleabilidade vetorial dos Transformers.

    Compara a diferença geométrica entre um vetor extraído de forma isolada 
    (Zero-Shot literal) e o vetor da mesma palavra extraído sob influência da 
    vizinhança sintática de uma sentença completa. A Distância do Cosseno é 
    utilizada para quantificar a distorção gerada pelo mecanismo de Auto-Atenção.

    Returns:
        None: Exibe o relatório analítico no console.
    """
    tok_yrl, mod_yrl, tok_pt, mod_pt = load_ai_ecosystem()
    
    amostras = [
        {"oracao": "taína-miri u-nheeng puranga", "alvo": "u-nheeng", "idioma": "Nheengatu", "tok": tok_yrl, "mod": mod_yrl},
        {"oracao": "a criancinha fala bonito", "alvo": "fala", "idioma": "Português", "tok": tok_pt, "mod": mod_pt},
        {"oracao": "apigá u-munuka imirá", "alvo": "imirá", "idioma": "Nheengatu", "tok": tok_yrl, "mod": mod_yrl},
        {"oracao": "o homem corta a árvore", "alvo": "árvore", "idioma": "Português", "tok": tok_pt, "mod": mod_pt}
    ]

    print("\n--- DIAGNÓSTICO MATEMÁTICO: O IMPACTO DO CONTEXTO ---")
    print("Métrica: Similaridade de Cosseno (1.0 = Nenhuma mudança / < 1.0 = O vetor se moveu)\n")

    for ex in amostras:
        idioma = ex["idioma"]
        alvo = ex["alvo"]
        oracao = ex["oracao"]
        tokenizer = ex["tok"]
        modelo = ex["mod"]

        # 1. Extração Estática: A palavra sem vizinhança sintática
        vetor_isolado = extract_contextual_vector(alvo, alvo, tokenizer, modelo)
        
        # 2. Extração Dinâmica: A palavra impregnada pelo mecanismo de Atenção
        vetor_contextual = extract_contextual_vector(oracao, alvo, tokenizer, modelo)

        if vetor_isolado is not None and vetor_contextual is not None:
            # Cálculo de Similaridade de Cosseno (Isometria)
            similaridade = F.cosine_similarity(vetor_isolado.unsqueeze(0), vetor_contextual.unsqueeze(0)).item()
            
            print(f"[{idioma}] Palavra: '{alvo}'")
            print(f"  -> Contexto injetado: '{oracao}'")
            print(f"  -> Similaridade (Isolado vs Contextual): {similaridade:.4f}")
            print(f"  -> Conclusão: O vetor sofreu uma distorção de {(1 - similaridade)*100:.1f}% devido à vizinhança sintática.\n")

if __name__ == "__main__":
    comparar_impacto_contexto()