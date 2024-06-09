import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from joblib import load
import tensorflow as tf


# Obtenha o diretório atual do arquivo Python
current_directory = os.path.dirname(os.path.abspath(__file__))

# Caminho para a pasta 'ia' a partir do diretório atual
ia_directory = os.path.join(current_directory, 'ia')

# Carregar o modelo
text_model = load_model(os.path.join(
    ia_directory, 'text_restoration_model.keras'))

maxlen = 33

# Carregar o modelo treinado
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
    input_sequence = tokenizer.texts_to_sequences([damaged_text])
    input_sequence = pad_sequences(
        input_sequence, maxlen=maxlen, padding='post')

    # Converter para tensores do TensorFlow
    input_sequence = tf.convert_to_tensor(input_sequence)

    # Previsão do modelo
    decoded_sequence = text_model.predict([input_sequence, input_sequence])
    decoded_sequence = np.argmax(decoded_sequence, axis=-1)[0]

    # Convertendo de volta para texto
    restored_text = ''.join(
        tokenizer.index_word[token] for token in decoded_sequence if token != 0)

    return region, restored_text
