  
import pytest
from mypkg.my_answers import evalPostfix



def test_evalPostfix(): 
  assert evalPostfix("55 44 + 3 *") == 297
