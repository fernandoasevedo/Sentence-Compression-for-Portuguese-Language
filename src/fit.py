from data.load_data import load_g1, load_seq_priberam
from model.fit import fit_model


if __name__ == '__main__':
    data = load_g1('data/G1-Pares/g1_pares_alignment_sentences.csv')
    fit_model(data)

    # pcsc = load_seq_priberam('data/PCSC-Pares/pcsc_alignment_sentences.csv')