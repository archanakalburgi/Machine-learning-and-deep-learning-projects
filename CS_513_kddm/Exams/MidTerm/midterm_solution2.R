rm(list=ls())

#################################################
#  name : Archana Kalburgi  
#  CWID : 10469491
# Mid-term Solution #2 
#################################################

rm(list=ls())

name <- file.choose()
dataSet <- read.csv(name, na.strings = c("?"))
dataSet[,1:7][dataSet[,1:7]=="?"] <- NA
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


# III Displaying the frequency table of "Infected" vs "MaritalStatus" 
myTable <- table(dataSet$Infected, dataSet$MaritalStatus)
ftable(myTable)


#IV Displaying Scatter plot of "Age", "MaritalStatus" and "MonthAtHospital", one pair at a time 
dataSet$MaritalStatus <- factor(dataSet$MaritalStatus, levels = c("Married", "Divorced", "Single"), labels = c(0, 1, 2)) 
pairs(dataSet[,c("Age", "MaritalStatus", "MonthAtHospital")])

plot(dataSet[,c("Age", "MaritalStatus")], main = "Age and MaritalStatus", pch = 10, col = 2)
plot(dataSet[,c("Age", "MonthAtHospital")], main = "Age and MonthAtHospital", pch = 10, col = 2)
plot(dataSet[,c("MaritalStatus", "MonthAtHospital")], main = "MaritalStatus And MonthAtHospital", pch = 10, col = 2)
plot(dataSet[,c("MaritalStatus", "Age")], main = "MaritalStatus And Age", pch = 18, col = 2)
plot(dataSet[,c("MonthAtHospital", "Age")], main = "MonthAtHospital And Age", pch = 10, col = 2)
plot(dataSet[,c("MonthAtHospital", "MaritalStatus")], main = "MonthAtHospital and MaritalStatus", pch = 10, col = 2)


# V.	Show box plots for columns:  “Age”, and “MonthAtHospital”
boxplot(dataSet[,c("Age", "MonthAtHospital")], main="Box Plot of Age And Months at the Hospital")


# VI.	Replacing the missing values of “Cases” with the “mean” of “Cases”.
x <- mean(dataSet$Cases, na.rm = TRUE)
dataSet[is.na(dataSet[,5]), 5] <- mean(dataSet$Cases, na.rm = TRUE)
