def isPalindrome(string):
    new_string = ""
    for s in string:
        if s.isalnum():
            new_string = new_string+s
    
    return new_string.lower() == new_string.lower()[::-1]

print(isPalindrome("A man, a plan, a canal: Panama")) 
print(isPalindrome("race a car"))


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
    
    parenthesis("", n, 0, 0) # 
    return ans  

print(generateParenthesis(0)) 
print(generateParenthesis(1))
print(generateParenthesis(2))
print(generateParenthesis(3))

'''
parenthesis("", 2, 0, 0):                                                           (7)
    if  
    else 
        0 < 2:
            parenthesis("(", 2, 1, 0)                                               (3)                                         
                                                                                
            parenthsis("(", 2, 1, 0):                                               
                if 
                else
                    1 < 2:
                        parenthesis("((", 2, 2, 0)                                  (2)

                        parenthesis("((", 2, 2, 0):
                            if 
                            else
                                2 = 2
                                2 > 0:
                                    parenthesis("(()", 2, 2, 1)                     (1)

                                    parenthesis("(()", 2, 2, 1)
                                        if
                                        else 
                                            2 = 2
                                            2 > 1:
                                                parenthesis("(())", 2, 2, 2)

                                                parenthesis("(())", 2, 2, 2):
                                                    if 4 = n*2:
                                                        ans = ["(())"] 


            parenthesis("(", 2, 1, 0):                                              (6)
                if
                else
                    if
                    left > right (1>0):
                        parenthesis("()", 2, 1, 1)                                  (5)
                            
                        parenthesis("()", 2, 1, 1):
                            if 
                            else
                                2 > 1:
                                    parenthesis("()(", 2, 2, 1)                     (4)

                                    parenthesis("()(", 2, 2, 1):        
                                        if
                                        else
                                        if
                                        2 > 1:
                                            parenthesis("()()", 2, 2, 2)

                                            parenthesis("()()", 2, 2, 2):
                                                ans = ["(())", "()()"]

'''