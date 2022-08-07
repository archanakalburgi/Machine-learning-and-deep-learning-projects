 #!/usr/bin/python

"""
Python  Functions
"""


# Write a Python function unique_list() that takes a list and returns a new list with unique elements of the first list.
# unique_list: (1,2,3,3,3,3,4,5)

def unique_list(aList):
    x = []
    for u in aList:
        if u not in x:
            x.append(u)  
    return x



#Write a Python function multiply()to multiply all giving numbers in a list.
#Sample List : (9, 2, 3, -5, 7)
#Expected Output : -1890
def multiply(aList):
    total = 1
    for i in aList:
        total = total*i
    return total  

# Write a function get_average() that takes an arbitrary number of arguments and returns the average of them
# Number of arguments : (5,6,8,10,21)
# Expected Output : 10
def get_average(*kargs):
    num = 0
    count = len(kargs)
    for k in kargs:
        num += k 

    return int(num/count)