import pandas as pd
from utils.dataloader import loader_tf
from utils.region_names import load_region_maps
from utils.alphabet import GreekAlphabet
import random
from typing import List, Tuple, Dict
import re


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """Carrega o dataset e retorna um DataFrame do pandas."""
    with open(dataset_path) as dataset_file:
        dataset = loader_tf(dataset_file=dataset_file)
    return pd.DataFrame(dataset)


def create_greek_alphabet(config: Dict) -> GreekAlphabet:
    alphabet_kwargs = dict(config['alphabet'])
    wordlist_path = alphabet_kwargs.pop('wordlist_path')
    with open(wordlist_path, 'r', encoding='utf-8') as f:
        alphabet = GreekAlphabet(wordlist_file=f, **alphabet_kwargs)
    return alphabet


def load_region_maps_from_files(main_path: str, sub_path: str) -> Dict[str, any]:
    region_map = {'main': None, 'sub': None}
    with open(main_path, 'r') as f:
        region_map['main'] = load_region_maps(f)
    with open(sub_path, 'r') as f:
        region_map['sub'] = load_region_maps(f)
    return region_map


def substitute_letters(word: str, removal_percentage: float, proximity_bias: float) -> Tuple[str, int]:
    num_chars_to_remove = int(len(word) * removal_percentage)
    characters = list(word)
    num_replacements = 0
    probabilities = [1.0] * len(characters)

    def adjust_probabilities(index: int):
        for i in range(len(probabilities)):
            if characters[i] != '_':
                distance = abs(i - index)
                if distance > 0:
                    probabilities[i] *= (1 + proximity_bias / distance)
                else:
                    probabilities[i] = 0

    while num_replacements < num_chars_to_remove:
        total_probability = sum(probabilities)
        normalized_probabilities = [
            p / total_probability for p in probabilities]
        index = random.choices(range(len(characters)),
                               weights=normalized_probabilities, k=1)[0]
        if characters[index] != '_':
            characters[index] = '_'
            num_replacements += 1
            adjust_probabilities(index)

    new_string = ''.join(characters)
    return new_string, num_replacements


def add_noise(word: str, noise_level: float) -> Tuple[str, int]:
    greek_alphabet = 'αβγδεζηθικλμνξοπρστυφχψω'
    characters = list(word)
    num_noise_chars = int(len(characters) * noise_level)
    noise_indices = random.sample(range(len(characters)), num_noise_chars)
    count_noise = 0

    for index in noise_indices:
        characters[index] = random.choice(greek_alphabet)
        count_noise += 1

    new_string = ''.join(characters)
    return new_string, count_noise


def damage_text_function(text: str, configs: Dict) -> Tuple[str, int, int]:
    removal_percentage = random.uniform(
        configs['min_removal_percentage'], configs['max_removal_percentage'])
    text_damaged, count_noise = add_noise(text, configs['noise_level'])
    text_damaged, count_replacements = substitute_letters(
        text_damaged, removal_percentage, configs['proximity_bias'])
    return text_damaged, count_replacements, count_noise


def substituir_palavras_randomicamente(texto, substituicao="___", proporcao=0.5):
    # Lista para armazenar as palavras substituídas
    palavras_substituidas = []

    # Função auxiliar para substituir palavras aleatoriamente
    def substitui_palavra(match):
        palavra = match.group(0)
        if random.random() < proporcao:
            palavras_substituidas.append(palavra)
            return substituicao
        else:
            return palavra

    # Substituir palavras no texto aleatoriamente
    texto_substituido = re.sub(r'\b\w+\b', substitui_palavra, texto)

    return texto_substituido, palavras_substituidas


def add_noise(texto):
    letras_alteradas = []
    novo_texto = ""
    for letra in texto:
        if letra.isalpha() and random.choice([True, False]):
            novo_texto += "_"
            letras_alteradas.append(letra)
        else:
            novo_texto += letra
    return novo_texto, letras_alteradas


def process_dataframe(df: pd.DataFrame, damage_text_configs: Dict) -> pd.DataFrame:
    df[['text_damaged', 'replacements']] = df['text'].apply(
        lambda x: pd.Series(add_noise(x))
    )
    return df


def export_dataframe(df: pd.DataFrame, file_path: str, file_format: str = 'csv'):
    if file_format == 'csv':
        df.to_csv(file_path, index=False)
    elif file_format == 'excel':
        df.to_excel(file_path, index=False)
    elif file_format == 'json':
        df.to_json(file_path, orient='records', lines=True)
    else:
        raise ValueError(
            "Formato de arquivo não suportado: {}".format(file_format))
