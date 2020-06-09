from data.utils import build_label, normalize_sent
import pandas as pd


def load_seq_priberam(data_path):
    print('Loading data')
    data = pd.read_csv(data_path, sep=',')
    data['tokens'] = data.original_sent.map(normalize_sent)
    data['labels'] = data.reduced_sent.map(normalize_sent)
    data['labels'] = data.apply(
        lambda row: build_label(row.tokens, row.labels),
        axis=1)
    data = data[['original_sent', 'tokens', 'labels']]
    data.columns = ['raw', 'sent', 'labels']
    return data


def load_g1(data_path):
    data = pd.read_csv(data_path, sep=',')
    data['tokens'] = data.original_sent.map(normalize_sent)
    data['labels'] = data.reduced_sent.map(normalize_sent)
    data['labels'] = data.apply(
        lambda row: build_label(row.tokens, row.labels),
        axis=1)
    data = data[['original_sent', 'tokens', 'labels']]
    data.columns = ['raw', 'sent', 'labels']
    return data
