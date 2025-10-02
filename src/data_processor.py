import os
import pymupdf
import re
import logging
from pathlib import Path

# Configuração básica do logging para exibir mensagens no console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extrai todo o texto de um único arquivo PDF usando um gerenciador de contexto."""
    try:
        with pymupdf.open(pdf_path) as doc:
            full_text = "".join(page.get_text("text") for page in doc)
        return full_text
    except Exception as e:
        logging.error(f"Erro ao ler o PDF {pdf_path.name}: {e}")
        return ""

def process_corpus(corpus_folder: str) -> str:
    """
    Processa todos os PDFs em uma pasta, normaliza e retorna um único texto concatenado.
    """
    corpus_path = Path(corpus_folder)
    if not corpus_path.is_dir():
        logging.error(f"A pasta do corpus '{corpus_folder}' não foi encontrada ou não é um diretório.")
        return ""
        
    all_texts = []
    pdf_files = list(corpus_path.glob('*.pdf'))

    if not pdf_files:
        logging.warning(f"Nenhum arquivo PDF encontrado na pasta '{corpus_folder}'.")
        return ""

    for pdf_path in pdf_files:
        logging.info(f"Processando: {pdf_path.name}")
        text = extract_text_from_pdf(pdf_path)
        if text:
            # 1. Junta palavras hifenizadas no final da linha.
            # 2. Substitui todas as sequências de espaços em branco (incluindo \n, \t) por um único espaço.
            normalized_text = re.sub(r'-\s*\n\s*', '', text)
            normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()
            all_texts.append(normalized_text)
            
    return " ".join(all_texts)