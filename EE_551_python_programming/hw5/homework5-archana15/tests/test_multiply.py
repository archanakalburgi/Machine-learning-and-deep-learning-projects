import pytest

from mypkg.my_answers import multiply

def test_multyply():
       assert multiply([9, 2, 3, -5, 7]) == -1890

