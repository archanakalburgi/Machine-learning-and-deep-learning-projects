"""
1. Twin Primes:

Twin prime numbers are a pair of prime number which differ by two, 
e.g: 3 and 5, 5 and 7, 11 and 13, and so on.

Write a program twin_prime.py which takes a number x as input and computes the maximum twin prime between 1 and x. 

A faster method to check if a number is prime is that it should not be divisible by any number between 2 to n.

Note: Your program should consider the case where x is a very large number.
"""
    
def prime_numbers(n):
    prime = [True for i in range(n + 2)]
    p = 2
    while p * p <= n + 1:
        if prime[p]:
            for i in range(p * 2, n + 2, p):
                prime[i] = False
        p += 1
    return prime

def twin_prime_numbers(x):
    prime = prime_numbers(x)
    result = list()
    for i in range(2, x-1):
        if prime[i] and prime[i + 2]:
            result.append((i, i+2))
    
    return result

# assert(twin_prime_numbers(15)==[(3, 5), (5, 7), (11, 13)])  
print(twin_prime_numbers(1000000))