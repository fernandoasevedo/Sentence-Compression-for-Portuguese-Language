from nltk.tokenize import wordpunct_tokenize
import logging


class Tokenizer(object):
    
    def __init__(self, name=None, doLower=True):
        self.name = 'Tokenizer' if name is None else name
        self.doLower = doLower

        logMessage('Tokenizer %s was crated doLower=%s' % (
            self.name, self.doLower))
 
    def tokenize(self, sentence):
        return self.simpleNorm(sentence).split()
    
    def simpleNorm(self, sentence):
        _sentence = sentence.strip()
        if self.doLower:
            return _sentence.lower()
        return _sentence


class DefaultTokenizer(Tokenizer):

    def __init__(self):
        Tokenizer.__init__(self, 'default_tokenizer')
    
    def tokenize(self, sentence):
        return wordpunct_tokenize(self.simpleNorm(sentence))


def logMessage(message:str):
    logging.info(message)


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
