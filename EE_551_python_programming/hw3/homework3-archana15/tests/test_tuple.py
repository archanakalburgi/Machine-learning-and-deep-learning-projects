import pytest

from mypkg.my_answers import tuple

x, y, z, f, m, n = tuple()


def test_element1():
    assert(x == 11)


def test_element2():
    assert(y == 1000)


def test_index():
    assert(z == 7)


def test_repeat():
    assert(f == 2)


def test_sum():
    assert(m == 10148)


def test_maximum():
    assert(n == 8679)
