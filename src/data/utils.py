from nltk.tokenize import wordpunct_tokenize


def log(message):
    print(message)


def normalize_sent(sent, stemmer=False):
    tokens = wordpunct_tokenize(sent)
    tokens = [t.lower() for t in tokens]

    return tokens


def build_label(original, reduced):
    labels = [1 for _ in original]
    idx = 0
    for i, t in enumerate(original):
        if t == reduced[idx]:
            labels[i] = 0
            idx += 1
        if idx >= len(reduced):
            break

    return labels
