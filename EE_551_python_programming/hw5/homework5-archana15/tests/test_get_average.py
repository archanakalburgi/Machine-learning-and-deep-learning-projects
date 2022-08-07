import pytest

from mypkg.my_answers import get_average


def test_get_average():
    assert get_average(5,6,8,10,21) == 10
