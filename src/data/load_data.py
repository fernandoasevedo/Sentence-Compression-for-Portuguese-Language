from data.utils import build_label, logMessage
import pandas as pd


def load_seq_priberam(data_path, textProcessor):
    logMessage('Loading data')
    data = pd.read_csv(data_path, sep=',')
    data['tokens'] = data.original_sent.map(textProcessor.tokenize)
    data['labels'] = data.reduced_sent.map(textProcessor.tokenize)
    data['labels'] = data.apply(
        lambda row: build_label(row.tokens, row.labels),
        axis=1)
    data = data[['original_sent', 'tokens', 'labels']]
    data.columns = ['raw', 'sent', 'labels']
    return data


def load_g1(data_path, textProcessor):
    data = pd.read_csv(data_path, sep=',')
    data['tokens'] = data.original_sent.map(textProcessor.tokenize)
    data['labels'] = data.reduced_sent.map(textProcessor.tokenize)
    data['labels'] = data.apply(
        lambda row: build_label(row.tokens, row.labels),
        axis=1)
    data = data[['original_sent', 'tokens', 'labels']]
    data.columns = ['raw', 'sent', 'labels']

    return data
