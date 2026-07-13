import re
import unicodedata
import pandas as pd

def normalize_unicode(text):
    """Converte o texto para a forma normal NFC."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    return unicodedata.normalize('NFC', text)

def clean_text_nheengatu(text):
    """
    Limpeza profunda preservando a estrutura do Nheengatu.
    1. Normalização NFC e Lowercase.
    2. Remoção de pontuação (exceto apóstrofo/glotal e hífen composicional).
    3. Remoção de espaços extras.
    """
    text = normalize_unicode(text).lower()
    
    # Regex: 
    # \w : letras e números
    # \s : espaços
    # '\- : apóstrofo e hífen (fonética e morfologia)
    # ,; : vírgula e ponto e vírgula (delimitadores para o Data Augmentation)
    text = re.sub(r"[^\w\s'\-,\;]", '', text)
    
    # Remover espaços múltiplos
    text = re.sub(r'\s+', ' ', text).strip()
    return text