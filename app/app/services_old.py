from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from joblib import load

import os

# Obtenha o diretório atual do arquivo Python
current_directory = os.path.dirname(os.path.abspath(__file__))

# Caminho para a pasta 'ia' a partir do diretório atual
ia_directory = os.path.join(current_directory, 'ia')

# Carregar o modelo treinado
text_model = load_model(os.path.join(ia_directory, 'text_model.h5'))
region_model = load_model(os.path.join(ia_directory, 'region_model.h5'))

# Carregar o LabelEncoder do arquivo
label_encoder = load(os.path.join(ia_directory, 'label_encoder.joblib'))

# Carregar o Tokenizer usado para preprocessar os dados de treinamento
with open(os.path.join(ia_directory, 'tokenizer.pkl'), 'rb') as f:
    tokenizer = pickle.load(f)

def restore_ancient_text(text):
    return predict_text_with_region(text)

# Função para prever a região
def predict_region(damaged_text):
    seq = tokenizer.texts_to_sequences([damaged_text.lower()])
    padded_seq = pad_sequences(seq, maxlen=100, padding='post')
    pred = region_model.predict(padded_seq)
    region_index = np.argmax(pred, axis=-1)[0]
    return label_encoder.inverse_transform([region_index])[0]

# Função para prever o texto com a região
def predict_text_with_region(damaged_text):
    region = predict_region(damaged_text)
    seq = tokenizer.texts_to_sequences([damaged_text.lower()])
    padded_seq = pad_sequences(seq, maxlen=100, padding='post')
    pred = text_model.predict(padded_seq)
    pred_indices = np.argmax(pred, axis=-1)
    # Remover padding (zeros) das previsões
    pred_indices = pred_indices[0]  # Considerar apenas a primeira sequência (batch size 1)
    pred_indices = [index for index in pred_indices if index != 0]  # Remover padding

    predicted_text = tokenizer.sequences_to_texts([pred_indices])
    return region, predicted_text[0]
