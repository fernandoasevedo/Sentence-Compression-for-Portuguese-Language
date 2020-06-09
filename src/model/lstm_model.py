from keras.layers import Dense, Dropout, Embedding, Input, LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Model


def init_model(seq_size, wv_size):
    '''
    Create a LSTM model based on the paper:
        Sentence Compression by Deletion with LSTMs
    '''

    # inputs = Input(shape=(seq_size, wv_size))
    inputs = Input(shape=(seq_size, 1))
    # inputs = Embedding(seq_size, 10, mask_zero=True)(inputs)

    a_lstm = LSTM(256, return_sequences=True)(inputs)
    a_dropout = Dropout(0.2)(a_lstm)

    b_lstm = LSTM(256, return_sequences=True)(a_dropout)
    b_dropout = Dropout(0.2)(b_lstm)

    c_lstm = LSTM(256, return_sequences=True)(b_dropout)
    predictions = TimeDistributed(
        Dense(2, activation='softmax'))(c_lstm)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    
    return model
