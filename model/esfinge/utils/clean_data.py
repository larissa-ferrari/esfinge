import re
import random


def clean_greek_text(text):
    """Função para limpeza de texto em grego"""
    # Remover espaços em branco extras no início e no fim
    text = text.strip()
    # Remover caracteres especiais (mantendo pontuações básicas e letras gregas)
    text = re.sub(r'[^α-ωΑ-Ωά-ώΆ-Ώ0-9 .,!?;:\']', '', text)
    # Normalizar o caso do texto (opcional, dependendo do caso de uso)
    text = text.lower()
    # Remover quebras de linha e tabulação
    text = text.replace('\n', ' ').replace('\t', ' ')
    # Remover espaços em branco extras no meio do texto
    text = re.sub(' +', ' ', text)
    return text


def random_swap(text):
    """Função para troca de palavras"""
    words = text.split()
    if len(words) < 2:
        return text
    idx1, idx2 = random.sample(range(len(words)), 2)
    words[idx1], words[idx2] = words[idx2], words[idx1]
    return ' '.join(words)


def add_noise(text, noise_level=0.1):
    altered_letters = []
    new_text = ""
    for letter in text:
        if letter.isalpha() and random.random() < noise_level:
            new_text += "_"
            altered_letters.append(letter)
        else:
            new_text += letter
    return new_text, altered_letters
