  
import pytest
from mypkg.my_answers import iterator



def test_iterator():
  i = [i for i in iterator(10)]
  assert (i) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
