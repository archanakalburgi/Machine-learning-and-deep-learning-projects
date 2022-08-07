import pytest

from mypkg.my_answers import isPalindrome


def test_isPalindrome():
    assert isPalindrome("A man, a plan, a canal: Panama") == True
    assert isPalindrome("race a car") == False
