rm(list=ls())

#################################################
#  name : Archana Kalburgi  
#  CWID : 10469491
#  Homework 2 
#################################################

rm(list=ls())

# Problem 1 : Load the "breast-cancer-wisconsin.data.csv" to R

name <- file.choose()
# Cancer <- read.csv("~/Downloads/breast-cancer-wisconsin.data.csv", na.strings = c("?"))
View(Cancer)
is.data.frame(Cancer)

# Problem 1 - I. Summarizing each column 

# calculating minimum, maximum and mean using apply() in-built function
minimum <- apply(Cancer, 2, min, na.rm = TRUE) 
minimum

maximum <- apply(Cancer, 2, max, na.rm = TRUE)
maximum

mean_of_data <- apply(Cancer, 2, mean, na.rm = TRUE)
mean_of_data

summary(Cancer)


# Problem 1 - II Identifying missing values 

# to find if there are any missing values
missing_values <- is.na(Cancer) 
missing_values

# to find which rows have the missing values
missing_in_rows = apply(is.na(Cancer), 1, sum) 
missing_in_rows

# to find which columns have how many missing values
missing_in_columns <- colSums(is.na(Cancer)) 
missing_in_columns

# to find total number of missing values in the data
missing_values_total <- sum(is.na(Cancer))
missing_values_total


# Problem 1 - III.	Replacing the missing values with the "mean" of the column.

  # 1. iterate through all the 11 columns using for loop
  # 2. calculate mean for every column
  # 3. replace N/A with mean 
for(columns in 1:11)
{
  mean_of_col <- mean(Cancer[,columns], na.rm = TRUE)
  Cancer[is.na(Cancer[,columns]), columns] <- mean_of_col
} 


# Problem 1 - IV.	Displaying the frequency table of "Class" vs. F6

required_table <- table(Cancer$Class, Cancer$F6)
# formatted table 
ftable(required_table)


# Problem 1 - V. Displaying the scatter plot of F1 to F6, one pair at a time.

# plotting scatter plots using plot in-built function 
plot(Cancer[2:7], main = "Scatter Plot from Column F1 to F6", pch = 20, col = "green") 


# Problem 1 - VI. Show histogram box plot for columns F7 to F9

# showing histogram using hist() in-built function  
hist(Cancer $F7, main = "Histogram plot for Column F7", xlab = "Column F7", col = "darkmagenta")
hist(Cancer $F8, main = "Histogram plot for Column F8", xlab = "Column F8", col = "light blue")
hist(Cancer $F9, main = "Histogram plot for Column F9", xlab = "Column F9", col = "darkgreen") 






# Problem 2 : 
# Delete all the objects from your R- environment. 
# Reload the "breast-cancer-wisconsin.data.csv" from canvas into R.
# Remove any row with a missing value in any of the columns.

# Delete all the objects from your R- environment. 
rm(list=ls())

# Reload the "breast-cancer-wisconsin.data.csv" from canvas into R.
name <- file.choose()
# Cancer <- read.csv("~/Downloads/breast-cancer-wisconsin.data.csv", na.strings = c("?"))


# Remove any row with a missing value in any of the columns.
Cancer <- na.omit(Cancer)
