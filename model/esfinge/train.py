import numpy as np
from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
import tensorflow as tf
from utils.data_loader import (
    load_dataset, create_greek_alphabet, load_region_maps_from_files,
    process_dataframe, export_dataframe
)
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed, Dropout
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed, Input, Bidirectional
from sklearn.preprocessing import LabelEncoder
import pickle
from joblib import dump, load
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from utils.clean_data import (
    clean_greek_text,
    random_swap,
    add_noise
)
import pandas as pd


# Configuração de Parâmetros
config = {
    'alphabet': {
        'wordlist_path': 'data/iphi-wordlist.txt',
        'wordlist_size': 35884,
    },
    'region_main_path': 'data/iphi-region-main.txt',
    'region_sub_path': 'data/iphi-region-sub.txt',
    'dataset_path': 'data/iphi.json',
}

# Carregar o dataset usando a função do módulo data_loader
df = load_dataset(config['dataset_path'])

# Selecionar apenas x% do DataFrame
df = df.sample(frac=0.001, random_state=1)

# Limpando dados
df['cleaned_text'] = df['text'].apply(clean_greek_text)

# Criar o Alfabeto Grego
alphabet = create_greek_alphabet(config)

# Carregar mapas de regiões
region_map = load_region_maps_from_files(
    config['region_main_path'], config['region_sub_path'])

# Configurações de Dano ao Texto
damage_text_configs = {
    'min_removal_percentage': 0.1,  # 0.1
    'max_removal_percentage': 0.3,  # 0.3
    'noise_level': 0.2,  # 0.2
    'proximity_bias': 1.0,
}


df[['text_damaged', 'replacements']] = df['cleaned_text'].apply(
    lambda x: pd.Series(add_noise(x))
)

data = df

# ----------------------------------------------------------------------------------------------
# Convertendo todo o texto para minúsculas
data['cleaned_text'] = [text.lower() for text in data['cleaned_text']]
data['text_damaged'] = [text.lower() for text in data['text_damaged']]
data['replacements'] = [[word.lower() for word in replacement]
                        for replacement in data['replacements']]
data['region_main'] = [region.lower() for region in data['region_main']]

# Juntando todo o texto para a tokenização
all_texts = data['cleaned_text'] + data['text_damaged'] + data['region_main']

# Instanciando o tokenizador
tokenizer = Tokenizer(filters='', char_level=True)
tokenizer.fit_on_texts(all_texts)

# Mapear os textos danificados e as regiões para sequências de inteiros
text_damaged_sequences = tokenizer.texts_to_sequences(data['text_damaged'])
region_sequences = tokenizer.texts_to_sequences(data['region_main'])

# Codificar as regiões
label_encoder = LabelEncoder()
region_labels = label_encoder.fit_transform(data['region_main'])

# Salvar o LabelEncoder em um arquivo
dump(label_encoder, 'label_encoder.joblib')

# Função para substituir as lacunas pelas palavras corretas
def create_target_sequences(damaged_seq, replacements, word_index):
    target_seq = np.array(damaged_seq)
    replacement_iter = iter(replacements)
    for i, token in enumerate(target_seq):
        if token == word_index['_']:
            replacement_word = next(replacement_iter)
            if replacement_word in word_index:
                target_seq[i] = word_index[replacement_word]
            else:
                raise KeyError(
                    f"Word '{replacement_word}' not found in tokenizer's word index.")
    return target_seq


# Criando as sequências de entrada (X) e as saídas (y)
X_text = []
y_text = []

for damaged_seq, replacement in zip(text_damaged_sequences, data['replacements']):
    target_seq = create_target_sequences(
        damaged_seq, replacement, tokenizer.word_index)
    X_text.append(damaged_seq)
    y_text.append(target_seq.tolist())

print()
print(data['text_damaged'].iloc[0])
print()
print(X_text[0])
print()
print(y_text[0])
print()

# Padding das sequências
max_len = max(max(len(seq) for seq in X_text), max(len(seq) for seq in y_text))
X_text_padded = pad_sequences(X_text, maxlen=max_len, padding='post')
y_text_padded = pad_sequences(y_text, maxlen=max_len, padding='post')

# Convertendo as saídas para categórico
y_text_categorical = [to_categorical(seq, num_classes=len(
    tokenizer.word_index) + 1) for seq in y_text_padded]
y_text_categorical = np.array(y_text_categorical)

# Padding das regiões
region_sequences_padded = pad_sequences(
    region_sequences, maxlen=max_len, padding='post')

# Definindo o modelo de classificação de região
region_model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1,
              output_dim=128, input_length=max_len),
    LSTM(128, return_sequences=False),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compilando o modelo de classificação de região
region_model.compile(
    optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinando o modelo de classificação de região
history_region_model = region_model.fit(
    X_text_padded, region_labels, epochs=100, batch_size=64)


# Gráficos
plot_model(region_model, to_file='region_model.png', show_shapes=True)

# summarize history for accuracy
plt.plot(history_region_model.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy_text_model.png')
# summarize history for loss
plt.plot(history_region_model.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss_text_model.png')

# Definindo o modelo de restauração de texto
text_model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1,
              output_dim=128, input_length=max_len),
    LSTM(256, return_sequences=True),
    Dropout(0.2),
    Dense(128, activation='relu'),
    LSTM(256, return_sequences=True),
    Dropout(0.2),
    Dense(128, activation='relu'),
    TimeDistributed(Dense(len(tokenizer.word_index) + 1, activation='softmax'))
])

# Compilando o modelo de restauração de texto
text_model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinando o modelo de restauração de texto
history_text_model = text_model.fit(
    X_text_padded, y_text_categorical, epochs=100, batch_size=32)

# Gráficos
plot_model(text_model, to_file='text_model.png', show_shapes=True)

# Resumir o histórico para precisão
plt.plot(history_text_model.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy_text_model.png')
# summarize history for loss
plt.plot(history_text_model.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss_text_model.png')

# Imprime o resumo do modelo
print("text_model")
text_model.summary()

print("region_model")
region_model.summary()

# Avaliar o modelo
loss, accuracy = text_model.evaluate(X_text_padded, y_text_categorical)
print("Text Model Loss:", loss)
print("Text Model Accuracy:", accuracy)

# Avaliar o modelo
loss, accuracy = region_model.evaluate(X_text_padded, region_labels)
print("Region Model Loss:", loss)
print("Region Model Accuracy:", accuracy)

# Supondo que você tenha treinado e compilado seu modelo e tenha uma referência a ele na variável 'model'
text_model.save('text_model.h5')

region_model.save('region_model.h5')

# Salvar o Tokenizer em um arquivo
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
