from data.load_data import load_g1, load_seq_priberam
from data.utils import DefaultTokenizer
from model.fit import fit_model

import os


if __name__ == '__main__':

    textProcessor = DefaultTokenizer()

    data = load_g1(
        os.path.join('data', 'G1-Pares', 'g1_pares_alignment_sentences.csv'),
        textProcessor)
    # fit_model(data)

    # pcsc = load_seq_priberam('data/PCSC-Pares/pcsc_alignment_sentences.csv')