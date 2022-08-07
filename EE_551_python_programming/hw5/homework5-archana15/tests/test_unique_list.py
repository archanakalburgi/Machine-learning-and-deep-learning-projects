import pytest

from mypkg.my_answers import unique_list


def test_unique_list():
    assert unique_list([1,2,3,3,3,3,4,5]) == [1, 2, 3, 4, 5]






