# Name : Archana Kalburgi
# CWID : 10469491
# Homework 3: KNN

remove(list=ls())

# loading and reading the csv file 
file<-file.choose()

cancer <- read.csv(file, na.strings = "?", colClasses=c("Class"="factor" ))
View(cancer)

is.factor(cancer$Class)

# removing the rows with missing values
omitted_cancer <- na.omit(cancer)

# creating test and train data 
mid <- sort (sample (nrow (omitted_cancer), round (.30*nrow(omitted_cancer))))

training <- omitted_cancer[-mid,]
test <- omitted_cancer[mid,] 


# install.packages("kknn")

library(kknn)


for(i in c(3,5,10))
{
  predict_k <- kknn (formula= Class~., training[,c(-1)] , test[,c(-1)], k=i, kernel ="rectangular"  )
  
  fit <- fitted(predict_k)
  
  wrong<- ( test$Class!=fit)
  rate<-sum(wrong)/length(wrong)
  rate
  print('--------------------------')
  print(i)
  print(table(test$Class,fit)) 
  print( rate)
}