from keras.utils import Sequence
from data.utils import logMessage
from data.utils import buildVocab
from model.constants import DEFAULT_TOKENS
from model.constants import EOS
from model.constants import BOS
from model.constants import UNKNOWN_TOKEN
from model.lstm_model import init_model

import numpy as np
import pickle


class DataGenerator(Sequence):

    def __init__(self, data, batch_size, vectorizer):
        self.data = data
        self.batch_size = batch_size
        self.vectorizer = vectorizer
    
    def __len__(self):
        '''Returns number of batches from data'''
        return int(np.floor(self.data.shape[0] / self.batch_size))

    def __getitem__(self, idx):
        'Generate one batch of data'
        bsize = self.batch_size
        X = np.array(
            self.data.sent.iloc[idx*bsize: (idx+1)*bsize] \
                .map(self.vectorizer.vectorize))

        Y = self.data.labels.iloc[idx*bsize: (idx+1)*bsize] \
                .map(self.vectorizer.vectorizerLabel)
        Y = np.array([[l for l in row] for row in Y])

        return np.array([row for row in X]), Y


class Vectorizer(object):

    def __init__(self, vocab, max_size=20):
        self.max_size = max_size
        self.vocab = vocab
        self.padding = 0
    
    def vectorize(self, sent):
        sent = sent[:self.max_size - 1]
        sent.append(EOS)
        encoded = np.zeros(self.max_size, dtype=np.int)
        for i, t in enumerate(sent):
            encoded[i] = self.vocab.get(t, UNKNOWN_TOKEN)
        return encoded

    def vectorizerLabel(self, labels):
        vec_labels = [0 for _ in range(self.max_size)]
        for i, l in enumerate(labels[:self.max_size]):
            vec_labels[i] = l
        return vec_labels


def fit_model(data, sent_size=20, batch_size=10):
    
    vocab = buildVocab(data.sent)

    vectorizer = Vectorizer(vocab, sent_size)

    data_reader = DataGenerator(data, batch_size, vectorizer)


    logMessage('Building data vectors')
    model = init_model(sent_size, len(vocab))
    logMessage(model.summary())

    model.fit_generator(
        generator=data_reader,
        epochs=3,
        verbose=1)

    logMessage('Saving model')
    model_json = model.to_json()
    with open("data/models/model.json", "w") as json_file:
        json_file.write(model_json)
    logMessage('Saving model weights')
    # serialize weights to HDF5
    model.save_weights("data/models/model.h5")

    logMessage('Saving vectorizer')
    with open('data/models/vec.pk', 'bw') as writer:
        pickle.dump(data_reader.vec, writer)