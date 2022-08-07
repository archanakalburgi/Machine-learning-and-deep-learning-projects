  
#!/usr/bin/python

"""
Object Oriented Programming and Python Classes
"""

"""
QUESTION 1: 
========================================================================================================
Write a class with name iterator which creates iterator type that iterates from 1 to a given limit. 
For example, if the limit is 10, then it prints  1 2 3 4 5 6 7 8 9 10. 

sample output = [1, 2 ,3 ,4 ,5 , 6, 7, 8, 9, 10] 

Hint: You could have three methods: __init__, __iter__, and _next_.
"""
class iterator:
    def __init__(self, limit):
        self.current = 0
        self.limit = limit
    def __iter__(self):
        return self
    def __next__(self):
        self.current += 1 
        if self.current <= self.limit:
            return self.current
        raise StopIteration
    
"""
QUESTION 2: 
========================================================================================================
Write a class with name unique_subsets to get all possible unique subsets from a set of distinct integers.
Example:
Input : [1, 2, 3]
output: [[], [3], [2], [2, 3], [1], [1, 3], [1, 2], [1, 2, 3]]

===========================
"""
class unique_subsets:
    def sub_sets(self, num):
        if num == []:
            return [[]]
        else:
            x, xs = num[0], num[1:] # [[]]
            sub = self.sub_sets(xs)
            return  sub + [s + [x] for s in sub]
            