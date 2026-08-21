import re
import unicodedata
import pandas as pd

def normalize_unicode(text):
    """
    Converte cadeias de caracteres para a Forma de Normalização Canonical (NFC).

    Args:
        text (str): Texto bruto de entrada.

    Returns:
        str: Texto normalizado de forma unificada, ou string vazia caso a entrada seja nula.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    return unicodedata.normalize('NFC', text)

def clean_text_nheengatu(text):
    """
    Executa a sanitização profunda respeitando as particularidades de línguas de baixos recursos.

    Aplica padronização morfológica através de normalização NFC, transição para lower case
    e aplicação de Regex restritivo, preservando caracteres alfanuméricos, espaços de 
    separação, apóstrofos (consoantes glotais) e hifens estruturais. Delimitadores 
    composicionais (vírgula e ponto e vírgula) são retidos para etapas de Data Augmentation.

    Args:
        text (str): Cadeia de texto bruta.

    Returns:
        str: Cadeia de texto sanitizada.
    """
    text = normalize_unicode(text).lower()
    
    # Filtro de purificação fonética e sintática
    text = re.sub(r"[^\w\s'\-,\;]", '', text)
    
    # Tratamento de distorções de espaçamento
    text = re.sub(r'\s+', ' ', text).strip()
    return text