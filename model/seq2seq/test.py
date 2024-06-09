from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import tensorflow as tf

# Carregar o modelo
model = load_model('text_restoration_model.keras')

with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

maxlen = 33


def restore_text(input_text):
    input_sequence = tokenizer.texts_to_sequences([input_text])
    input_sequence = pad_sequences(
        input_sequence, maxlen=maxlen, padding='post')

    # Converter para tensores do TensorFlow
    input_sequence = tf.convert_to_tensor(input_sequence)

    # Previsão do modelo
    decoded_sequence = model.predict([input_sequence, input_sequence])
    decoded_sequence = np.argmax(decoded_sequence, axis=-1)[0]

    # Convertendo de volta para texto
    restored_text = ''.join(
        tokenizer.index_word[token] for token in decoded_sequence if token != 0)
    return restored_text


# Exemplo de uso
new_text = "Οὐκ ἔ_τιν ἄνθρωπος ἀνὴρ _οφός."
restored_text = restore_text(new_text)
print(f'Restored text: {restored_text}')
