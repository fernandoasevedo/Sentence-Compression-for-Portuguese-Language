from data.utils import build_label


def test_buildLabel_Equal():
    source = 'isso é um teste'.split()
    reduced = 'isso é um teste'.split()
    expected = [0] * len(source)
    assert expected == build_label(source, reduced)


def test_buildLabel_RemoveFromLeft():
    source = 'isso é um teste'.split()
    reduced = 'é um teste'.split()
    expected = [0] * len(source)
    expected[0] = 1
    assert expected == build_label(source, reduced)


def test_buildLabel_RemoveFromRight():
    source = 'isso é um teste'.split()
    reduced = 'isso é um'.split()
    expected = [0] * len(source)
    expected[-1] = 1
    assert expected == build_label(source, reduced)


def test_buildLabel_RemoveFromMidle():
    source = 'isso é um teste'.split()
    reduced = 'isso teste'.split()
    expected = [0] * len(source)
    expected[1] = 1
    expected[2] = 1
    assert expected == build_label(source, reduced)