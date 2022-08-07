import pytest
from mypkg.my_answers import LinkedList
from mypkg.my_answers import countNodes



def test_countNodes():
  linkedList = LinkedList()
  linkedList.add(1)
  linkedList.add(2)
  linkedList.add(3)
  linkedList.add(4)
  linkedList.add(5)
  assert (countNodes(linkedList.head)) == 5
