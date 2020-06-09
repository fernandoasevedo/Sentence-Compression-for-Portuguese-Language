from keras.utils import Sequence
from model.lstm_model import init_model
import numpy as np
import pickle


def log(message):
    print(message)


class DataGenerator(Sequence):

    def __init__(self, size, data, batch_size):
        self.size = size
        self.data = data
        self.batch_size = batch_size
        self.__build_vocab()
        self.vec = Vectorizer(self.vocab, size)
    
    def __len__(self):
        '''Returns number of batches from data'''
        return int(np.floor(self.data.shape[0] / self.batch_size))

    def __getitem__(self, idx):
        'Generate one batch of data'
        bsize = self.batch_size
        X = np.array(
            self.data.sent.iloc[idx*bsize: (idx+1)*bsize] \
                .map(self.vec.vectorize))

        Y = self.data.labels.iloc[idx*bsize: (idx+1)*bsize] \
                .map(self.vec.label_vector)
        Y = np.array([[l for l in row] for row in Y])
        return np.array([row for row in X]), Y

    def __build_vocab(self):
        self.vocab = dict()
        self.vocab['<eos>'] = 1
        self.vocab['<sta>'] = 2
        self.vocab['<unk>'] = 3
        for sent in self.data.sent:
            for t in sent:
                self.vocab[t] = self.vocab.get(
                    t,
                    len(self.vocab) + 1)

class Vectorizer(object):

    def __init__(self, vocab, max_size=20):
        self.eos = '<eos>'
        self.start = '<sta>'
        self.unkown = '<unk>'
        self.max_size = max_size
        self.vocab = vocab
        self.padding = 0
        self.N = len(self.vocab) + 1
    
    def vectorize(self, sent):
        sent = sent[:self.max_size - 1]
        sent.append('<eos>')
        unk = self.vocab[self.unkown]
        '''
        encoded = np.zeros((self.max_size, self.N), dtype=np.bool)
        for i, t in enumerate(sent):
            idx = self.vocab.get(t, unk)
            encoded[i][idx] = 1
        '''
        encoded = np.zeros((self.max_size, 1), dtype=np.int)
        for i, t in enumerate(sent):
            encoded[i][0] = self.vocab.get(t, unk)
        return encoded

    def label_vector(self, labels):
        vec_labels = [[0] for _ in range(self.max_size)]
        for i, l in enumerate(labels[:self.max_size]):
            vec_labels[i][0] = l
        return vec_labels


def fit_model(data):
    size = 20
    data_reader = DataGenerator(size, data, 10)    
    log('Building data vectors')    
    model = init_model(size, len(data_reader.vocab) + 1)
    log(model.summary())
    model.fit_generator(
        generator=data_reader,
        epochs=10,
        verbose=1)

    log('Saving model')
    model_json = model.to_json()
    with open("data/models/model.json", "w") as json_file:
        json_file.write(model_json)
    log('Saving model weights')
    # serialize weights to HDF5
    model.save_weights("data/models/model.h5")

    log('Saving vectorizer')
    with open('data/models/vec.pk', 'bw') as writer:
        pickle.dump(data_reader.vec, writer)