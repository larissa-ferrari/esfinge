from utils.data_loader import (
    load_dataset
)
from utils.clean_data import (
    clean_greek_text,
    random_swap,
    add_noise
)
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Masking
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

# Carregar o dataset usando a função do módulo data_loader
df = load_dataset('data/iphi.json')

# Selecionar apenas x% do DataFrame
df = df.sample(frac=0.0001, random_state=1)

# Limpando dados
df['cleaned_text'] = df['text'].apply(clean_greek_text)

# Criando o DataFrame
df = pd.DataFrame(data)

# Aplicar a técnica de troca de palavras
df[['text_damaged', 'replacements']] = df['cleaned_text'].apply(
    lambda x: pd.Series(add_noise(x))
)

print(df.head())

# Função para tokenização
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(df['text_damaged'] + df['cleaned_text'])

# Convertendo textos para sequências
sequences_damaged = tokenizer.texts_to_sequences(df['text_damaged'])
sequences_restored = tokenizer.texts_to_sequences(df['cleaned_text'])

# Padding das sequências
max_sequence_length = max(max(len(seq) for seq in sequences_damaged), max(
    len(seq) for seq in sequences_restored))

# Definindo o valor de maxlen
print(f'Max sequence length: {max_sequence_length}')

X = pad_sequences(sequences_damaged,
                  maxlen=max_sequence_length, padding='post')
y = pad_sequences(sequences_restored,
                  maxlen=max_sequence_length, padding='post')

# Definindo parâmetros
num_tokens = len(tokenizer.word_index) + 1  # +1 para o token zero
embedding_dim = 100
latent_dim = 256

# Dividindo os dados em treino e validação
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Preparação dos dados de saída para o decodificador
y_train = np.expand_dims(y_train, -1)
y_val = np.expand_dims(y_val, -1)

# Modelo Seq2Seq com LSTM
# Codificador
encoder_inputs = Input(shape=(max_sequence_length,))
encoder_embedding = Embedding(
    num_tokens, embedding_dim, mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decodificador
decoder_inputs = Input(shape=(max_sequence_length,))
decoder_embedding = Embedding(
    num_tokens, embedding_dim, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(
    decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(num_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Modelo
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinamento do modelo
history_text_restoration_model = model.fit([X_train, X_train], y_train, batch_size=64,
                                           epochs=100, validation_data=([X_val, X_val], y_val))

# Avaliação do modelo
loss, accuracy = model.evaluate([X_val, X_val], y_val)
print(f'Loss: {loss}, Accuracy: {accuracy}')

model.save('text_restoration_model.keras')

# Salvando o tokenizer
with open('tokenizer.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)


# Gráficos
plot_model(model, to_file='text_restoration_model.png', show_shapes=True)

# Resumir o histórico para precisão
plt.plot(history_text_restoration_model.history['accuracy'])
plt.plot(history_text_restoration_model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy_text_restoration_model.png')
# Resumir o histórico de perdas
plt.plot(history_text_restoration_model.history['loss'])
plt.plot(history_text_restoration_model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss_text_restoration_model.png')
