#!/usr/bin/python

"""
Python Functions and Recursions
"""
"""
QUESTION 1:  
========================================================================================================
Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.
For example, given n = 3, a solution set is:
["((()))","(()())","(())()","()(())","()()()"]
Write a function named generateParenthesis that takes an integer as an input and returns a list of strings 
as an output. Note that you can define a function inside a function if necessary.
"""
def generateParenthesis(n):
    ans = [] 
    def parenthesis(string , n, countLeft, countRight):
        if len(string) == n * 2: # base case 
            ans.append(string)
        
        else:
            if countLeft < n:
                parenthesis(string + "(", n, countLeft + 1, countRight)
            
            if countLeft > countRight:
                parenthesis(string + ")", n, countLeft, countRight + 1) 
    
    parenthesis("", n, 0, 0) 
    return ans 


"""
QUESTION 2: 
========================================================================================================
Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.
Note: For the purpose of this problem, we define empty string as valid palindrome.
Example 1:
========================================
Input: "A man, a plan, a canal: Panama"
Output: true
Example 2:
=========================================
Input: "race a car"
Output: false
Write a function named isPalindrome that takes a string as an input and returns a bool as an output.
"""

def isPalindrome(string):
    new_string = ""
    for s in string:
        if s.isalnum():
            new_string = new_string+s
    
    return new_string.lower() == new_string.lower()[::-1]
