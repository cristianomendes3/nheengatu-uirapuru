import torch
from transformers import AutoTokenizer, AutoModel

MODELS = {
    "yrl": "dominguesm/canarim-bert-nheengatu",
    "pt": "neuralmind/bert-base-portuguese-cased"
}

def load_ai_ecosystem():
    print("[INFO] Carregando ecossistema de Inteligência Artificial...")
    tok_yrl = AutoTokenizer.from_pretrained(MODELS["yrl"])
    mod_yrl = AutoModel.from_pretrained(MODELS["yrl"]).eval()
    
    tok_pt = AutoTokenizer.from_pretrained(MODELS["pt"])
    mod_pt = AutoModel.from_pretrained(MODELS["pt"]).eval()
    return tok_yrl, mod_yrl, tok_pt, mod_pt

def extract_contextual_vector(sentence, target_word, tokenizer, model):
    """Realiza o Offset Mapping e extrai o vetor isolado impregnado de contexto."""
    char_start = sentence.find(target_word)
    if char_start == -1:
        return None
    char_end = char_start + len(target_word)
    
    encoding = tokenizer(
        sentence, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=True
    )
    
    offsets = encoding["offset_mapping"][0].tolist()
    
    # Isola os sub-tokens que pertencem à palavra-alvo
    target_indices = []
    for idx, (start, end) in enumerate(offsets):
        # Ignora tokens especiais (onde start e end são 0)
        if start == end: continue 
        if start < char_end and end > char_start:
            target_indices.append(idx)
            
    if not target_indices:
        return None
        
    with torch.no_grad():
        outputs = model(encoding["input_ids"])
        hidden_states = outputs.last_hidden_state[0] # Formato: [num_tokens, 768]
        
    # Mean Pooling da palavra
    word_vector = torch.mean(hidden_states[target_indices], dim=0)
    return word_vector

if __name__ == "__main__":
    tok_yrl, mod_yrl, tok_pt, mod_pt = load_ai_ecosystem()
    
    # Os 5 exemplos estruturados (O Padrão Ouro que a Equipe de Letras nos enviará)
    amostras = [
        {"oracao_yrl": "taína-miri u-nheeng puranga", "alvo_yrl": "u-nheeng", "oracao_pt": "a criancinha fala bonito", "alvo_pt": "fala"},
        {"oracao_yrl": "apigá u-munuka imirá", "alvo_yrl": "imirá", "oracao_pt": "o homem corta a árvore", "alvo_pt": "árvore"},
        {"oracao_yrl": "amana u-ri kuese", "alvo_yrl": "amana", "oracao_pt": "a chuva veio ontem", "alvo_pt": "chuva"},
        {"oracao_yrl": "aracú piré puranga", "alvo_yrl": "piré", "oracao_pt": "o aracu é um peixe bonito", "alvo_pt": "peixe"},
        {"oracao_yrl": "yané ruka iwate", "alvo_yrl": "ruka", "oracao_pt": "nossa casa é alta", "alvo_pt": "casa"}
    ]

    print("\n--- INICIANDO TESTE BILÍNGUE DE OFFSET MAPPING ---")
    for i, ex in enumerate(amostras):
        vec_yrl = extract_contextual_vector(ex["oracao_yrl"], ex["alvo_yrl"], tok_yrl, mod_yrl)
        vec_pt = extract_contextual_vector(ex["oracao_pt"], ex["alvo_pt"], tok_pt, mod_pt)
        
        print(f"\nExemplo [{i+1}]")
        print(f"  Nheengatu: '{ex['alvo_yrl']}' dentro de '{ex['oracao_yrl']}' -> Tensor de {vec_yrl.shape[0]} dimensões capturado.")
        print(f"  Português: '{ex['alvo_pt']}' dentro de '{ex['oracao_pt']}' -> Tensor de {vec_pt.shape[0]} dimensões capturado.")