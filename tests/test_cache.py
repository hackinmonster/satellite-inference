from cache import _key


def test_same_bytes_same_key():
    assert _key(b"abc") == _key(b"abc")


def test_different_bytes_different_key():
    assert _key(b"abc") != _key(b"abd")


def test_key_is_hex():
    k = _key(b"hello")
    assert k.startswith("sat:")
    assert len(k) == 4 + 64
