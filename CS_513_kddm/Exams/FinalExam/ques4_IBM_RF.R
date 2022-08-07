# Name : Archana Kalburgi 
# CWID : 10469491
# Problem #4 Solution -> Random Forest classification 

remove(list=ls())

# Load the “IBM_Attrition_v3.csv”. 
ibm <-read.csv("/Users/archanakalburgi/Downloads/IBM_Attrition_v3.csv", na.strings = "?" , stringsAsFactors=TRUE)
# View(ibm)

# Read all the columns as numeric variables except the “Attrition”,  “JobSatisfaction” and “MaritalStatus” columns
ibm$Age = as.numeric(ibm$Age)
ibm$JobSatisfaction = factor(ibm$JobSatisfaction)
ibm$MaritalStatus = factor(ibm$MaritalStatus)
ibm$MonthlyIncome = as.numeric(ibm$MonthlyIncome)
ibm$YearsAtCompany = as.numeric(ibm$YearsAtCompany)
ibm$Attrition = factor(ibm$Attrition, levels = c('Yes','No'),labels = c(1,2))

# a) Delete rows with missing values 
ibm_clean = na.omit(ibm)
# View(ibm_clean)

# b) Select 30% of the records as the test dataset and the remaining records as the training dataset
index<-sort(sample(nrow(ibm_clean),round(.30*nrow(ibm_clean))))
training<- ibm_clean[-index,]
test<- ibm_clean[index,]

# c)	Perform Random Forest classification for the “attrition” column  
# install.packages('randomForest')
library(randomForest)
fit <- randomForest(Attrition~., data=training[,0:6], importance=TRUE, ntree=100)
importance(fit)

varImpPlot(fit)
dev.off()

# d)	Score the test dataset
Prediction <- predict(fit, test[,0:6])
table(actual=test$Attrition,Prediction)

# e)	Measure the error rate. 
wrong<- (test$Attrition!=Prediction )
error_rate<-sum(wrong)/length(wrong)
error_rate

remove(list=ls())