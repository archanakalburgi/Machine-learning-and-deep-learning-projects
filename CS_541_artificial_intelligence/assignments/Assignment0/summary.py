"""
3. Data Summary

Write a program summary.py that computes the minimum, maximum, mean
and standard deviation of the number of COVID cases over time for a state
(input by the user).

The csv file for the COVID data can be found here.

"""

import math
import pandas as pd

def data_summary():
    df = pd.read_csv("us-states.csv")

    df["state"].isnull().sum()
    
    input_state = input("Enter state : ")

    df_state = df[(df["state"]==input_state)]
    
    maximum = df_state["cases"].max()
    
    minimum = df_state["cases"].min()
    
    mean = sum(df_state["cases"])/len(df_state)

    temp = sum((df_state["cases"] - mean)**2) 
    standard_deviation = math.sqrt(temp/(len(df_state)-1))

    print(f"Maximum : {maximum}")
    print(f"Minimum : {minimum}")
    print(f"Mean : {mean}")
    print(f"Standard deviation : {standard_deviation}")

data_summary()