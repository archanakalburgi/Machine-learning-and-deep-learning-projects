
# Question 1 - Load the "breast-cancer-wisconsin.data.csv" from canvas into R and perform the EDA analysis by:
dataSet <- read.csv("breast-cancer-wisconsin.data.csv", na.string = "?")
dataSet[,1:11][dataSet[,1:11]=="?"] <- NA


#To display the dataset that we've loaded
View(dataSet)


# I.	Summarizing each column (e.g. min, max, mean )
MinAll <- apply(dataSet, 2, min, na.rm = TRUE) 
MinAll

MaxAll <- apply(dataSet, 2, max, na.rm = TRUE)
MaxAll

MeanAll <- apply(dataSet, 2, mean, na.rm = TRUE)
MeanAll 

summary(dataSet)


# II.	Identifying missing values
missingValuesSum <- sum(is.na(dataSet))
missingValuesSum

missingValuesColSum <- colSums(is.na(dataSet))
missingValuesColSum


# III.	Replacing the missing values with the "mean" of the column.
for(i in 1:ncol(dataSet)){
  dataSet[is.na(dataSet[,i]), i] <- mean(dataSet[,i], na.rm = TRUE)
}

View(dataSet)

# IV.	Displaying the frequency table of "Class" vs. F6
myTable <- table(dataSet$Class, dataSet$F6)
ftable(myTable)


# V.	Displaying the scatter plot of F1 to F6, one pair at a time
plot(dataSet[2:7], main = "Scatter Plot of F1 to F6", pch = 10, col = 2)


# VI.	Show histogram box plot for columns F7 to F9

hist(dataSet$F7, main = "Histogram Box Plot for Column F7", col = 2)
hist(dataSet$F8, main = "Histogram Box Plot for Column F8", col = 2)
hist(dataSet$F9, main = "Histogram Box Plot for Column F9", col = 2)


# Question no 2
# Delete all the objects from your R- environment. 
rm(list=ls())
   

# Reload the "breast-cancer-wisconsin.data.csv" from canvas into R.
dataSet <- read.csv("breast-cancer-wisconsin.data.csv", na.string = "?")
dataSet[,1:11][dataSet[,1:11]=="?"] <- NA


# Remove any row with a missing value in any of the columns.
dataSet <- na.omit(dataSet)

