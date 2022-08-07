#!/usr/bin/python

"""
Python Core object Types
"""


def tuple():
    """
    This is to review tuples and basic operations.
    """
    # Assign the first element of the tuple "t1" to x
    t1 = (11, 100, 99, 1000, 999)

    x = t1[0]

    # Print the second from the last element of the tuple "t1".
    y = t1[-2]
    print(y)
    
    #What's the index of 2 in the tuple "t2"?
    t2 = (55, 777, 54, 6, 76, 101, 1, 2, 8679, 99, 123, 99, 76)
    z = t2.index(2)

    #How many times does 99 occur in the tuple "t2"?
    f = t2.count(99)

    # What is the sum of all the numbers in the tuple "t2"?
    
    m = sum(t2)

    # What is the maximum value in the tuple "t2"?
    n = max(t2)

    return x, y, z, f, m, n



tuple()




