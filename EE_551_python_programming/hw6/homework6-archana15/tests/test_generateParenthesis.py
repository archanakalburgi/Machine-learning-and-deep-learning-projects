import pytest

from mypkg.my_answers import generateParenthesis

def test_generateParenthesis():
    assert generateParenthesis(0) == ['']
    assert generateParenthesis(1) == ['()']
    assert generateParenthesis(2) == ['(())', '()()']
    assert generateParenthesis(3) == ['((()))', '(()())', '(())()', '()(())', '()()()']
   

