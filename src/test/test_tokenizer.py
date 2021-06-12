from data.utils import Tokenizer
from data.utils import DefaultTokenizer


def test_defaultTokenizer():
    tokenizer = Tokenizer()
    test = 'isso, é apenas um Teste.'
    expected = test.lower().split()
    assert expected == tokenizer.tokenize(test)


def test_NLTKTokenizer():
    tokenizer = DefaultTokenizer()
    test = 'isso, é apenas um Teste.'
    expected = 'isso , é apenas um teste .'.split()
    assert expected == tokenizer.tokenize(test)


def test_diffDefaultAndNLTK():
    tokenizer = Tokenizer()
    nltkTtokenizer = DefaultTokenizer()
    test = 'isso, é apenas um Teste.'
    assert nltkTtokenizer.tokenize(test) != tokenizer.tokenize(test)

