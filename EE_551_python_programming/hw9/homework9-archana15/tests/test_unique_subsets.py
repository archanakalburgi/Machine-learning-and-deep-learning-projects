import pytest
from mypkg.my_answers import unique_subsets



def test_unique_subsets():
    		assert (unique_subsets().sub_sets([1,2,3])) == [[], [3], [2], [2, 3], [1], [1, 3], [1, 2], [1, 2, 3]]
