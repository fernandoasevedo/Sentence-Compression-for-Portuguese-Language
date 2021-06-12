from keras.layers import Dense, Dropout, Embedding, Input, LSTM, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.models import Model

import numpy as np


def init_model(seq_size, wv_size, hidden_size=256, emb_dim=100):
    '''
    Create a LSTM model based on the paper:
        Sentence Compression by Deletion with LSTMs
    '''
    embedding_matrix = np.random.uniform(size=(wv_size, emb_dim))
    embedding_layer = Embedding(
        wv_size, emb_dim, input_length=seq_size,
        weights=[embedding_matrix],
        trainable=True,
        mask_zero=True)

    encoder_inputs_placeholder = Input(shape=(seq_size,))
    x = embedding_layer(encoder_inputs_placeholder)

    a_lstm, h, c = LSTM(
        hidden_size,
        return_state=True,
        return_sequences=True)(x)
    enconder_stats = [h, c]
    
    a_dropout = Dropout(0.2)(a_lstm)

    b_lstm = LSTM(hidden_size, return_sequences=True)(a_dropout)
    b_dropout = Dropout(0.2)(b_lstm)

    c_lstm = LSTM(
        hidden_size,
        return_sequences=True,)(b_dropout)
    predictions = TimeDistributed(
        Dense(2, activation='softmax'))(c_lstm)

    model = Model(inputs=encoder_inputs_placeholder, outputs=predictions)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    
    return model
