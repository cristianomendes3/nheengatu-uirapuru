import torch
import warnings
import transformers
from transformers import AutoTokenizer, AutoModel

# Supressão de verbosidades de diagnóstico da API do Hugging Face
transformers.logging.set_verbosity_error()

# Supressão de avisos genéricos de compilação
warnings.filterwarnings("ignore")

MODELS = {
    "yrl": "dominguesm/canarim-bert-nheengatu",
    "pt": "neuralmind/bert-base-portuguese-cased"
}

def load_ai_ecosystem():
    """
    Instancia o ecossistema bilíngue de Transformers.

    Carrega as arquiteturas pré-treinadas (Canarim-BERT para Nheengatu e
    BERTimbau para Português), alocando tanto os tokenizadores quanto
    os grafos de inferência diretamente na memória RAM em modo de avaliação (.eval()).

    Returns:
        tuple: (tokenizer_yrl, model_yrl, tokenizer_pt, model_pt)
    """
    print("[INFO] Carregando ecossistema de Inteligência Artificial...")
    tok_yrl = AutoTokenizer.from_pretrained(MODELS["yrl"])
    mod_yrl = AutoModel.from_pretrained(MODELS["yrl"]).eval()
    
    tok_pt = AutoTokenizer.from_pretrained(MODELS["pt"])
    mod_pt = AutoModel.from_pretrained(MODELS["pt"]).eval()
    return tok_yrl, mod_yrl, tok_pt, mod_pt

def extract_contextual_vector(sentence, target_word, tokenizer, model):
    """
    Mapeia os offsets de caracteres e realiza a extração hiperdimensional da palavra-alvo.

    Localiza fisicamente a palavra dentro da string matriz, processa a sentença
    inteira pelo Transformer (absorvendo contexto bidirecional) e realiza
    Mean Pooling (Média Tensional) estritamente sobre os sub-tokens da palavra-alvo.

    Args:
        sentence (str): Sentença base fornecedora do contexto sintático.
        target_word (str): Palavra-alvo a ser isolada e extraída matematicamente.
        tokenizer (AutoTokenizer): Tokenizador WordPiece do idioma correspondente.
        model (AutoModel): Modelo de linguagem Transformer do idioma correspondente.

    Returns:
        torch.Tensor or None: Tensor unidimensional de 768 dimensões ou None se falhar o pareamento.
    """
    char_start = sentence.lower().find(target_word.lower())
    if char_start == -1:
        return None
    char_end = char_start + len(target_word)
    
    encoding = tokenizer(
        sentence, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=True
    )
    
    offsets = encoding["offset_mapping"][0].tolist()
    
    # Identificação rigorosa dos sub-tokens baseados na indexação do WordPiece
    target_indices = []
    for idx, (start, end) in enumerate(offsets):
        # Descarte de tokens funcionais internos do modelo ([CLS], [SEP])
        if start == end: continue 
        if start < char_end and end > char_start:
            target_indices.append(idx)
            
    if not target_indices:
        return None
        
    with torch.no_grad():
        outputs = model(encoding["input_ids"])
        hidden_states = outputs.last_hidden_state[0] # Formato matricial latente: [num_tokens, 768]
        
    # Condensamento matricial (Mean Pooling) para restaurar a integridade da palavra fragmentada
    word_vector = torch.mean(hidden_states[target_indices], dim=0)
    return word_vector

if __name__ == "__main__":
    tok_yrl, mod_yrl, tok_pt, mod_pt = load_ai_ecosystem()
    
    # Amostragem padrão ouro validada linguisticamente
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