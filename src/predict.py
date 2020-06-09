from keras.models import model_from_json
from nltk.tokenize import wordpunct_tokenize
import numpy as np
import pickle

import sys


def decode(output, sentences):
    labels = [
        [np.argmax(token) for token in sent]
        for sent in output
    ]

    return [
        [token for j, token in enumerate(sent) if labels[i][j] == 0]
        for i, sent in enumerate(sentences)
    ]


def load_model():
    print('Loading vec')
    with open('data/models/vec.pk', 'br') as reader:
        vec = pickle.load(reader)

    print('Loading model')
    with open('data/models/model.json') as json_file:
        model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights("data/models/model.h5")

    return model, vec

def predict(model, vec, sentences):
    
    vec_sent = [vec.vectorize(sent) for sent in sentences]
    vec_sent = np.array(vec_sent)
    
    labels = model.predict(vec_sent)
    print(decode(labels, sentences))


if __name__ == '__main__':

    model, vec = load_model()

    if len(sys.argv) > 1:
        with open(sys.argv[1]) as reader:
            sentences = reader.readlines()
    else:
        print('Input your sentence')
        sentences = input()
        sentences = [sentences]

    sentences = [
        wordpunct_tokenize(sent.lower())
        for sent in sentences]
    print(sentences)
    predict(model, vec, sentences)