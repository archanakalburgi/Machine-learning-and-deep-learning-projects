#####################################
# Name : Archana Kalburgi
# CWID : 10469491
# Assignment 6.2
####################################

rm(list=ls())

# loading data into R
file<-file.choose()
bc <- read.csv(file) 

# install.packages('randomForest')
library(randomForest)

set.seed(123)

# remove all the records with missing value
bc_clean <- na.omit(bc)
bc_clean <- bc_clean[2:11]

set.seed(111)

# factorizing all the columns 
bc_clean$F1 = factor(bc_clean$F1)
bc_clean$F2 = factor(bc_clean$F2)
bc_clean$F3 = factor(bc_clean$F3) 
bc_clean$F4 = factor(bc_clean$F4) 
bc_clean$F5 = factor(bc_clean$F5) 
bc_clean$F6 = factor(bc_clean$F6)
bc_clean$F7 = factor(bc_clean$F7) 
bc_clean$F8 = factor(bc_clean$F8) 
bc_clean$F9 = factor(bc_clean$F9) 
bc_clean$Class = factor(bc_clean$Class) 

# 30% test data , remaining as test data 
index <- sort(sample(nrow(bc_clean),round(.30*nrow(bc_clean))))
training <- bc_clean[-index,]
test <- bc_clean[index,]

# Random Forest 
fit <- randomForest(Class~., data=training, importance=TRUE, ntree=1000)
importance(fit)
varImpPlot(fit)
Prediction <- predict(fit, test)
table(actual=test[,10],Prediction)

# calculating error rate 
wrong<- (test[,10]!=Prediction )
error_rate<-sum(wrong)/length(wrong)
error_rate 