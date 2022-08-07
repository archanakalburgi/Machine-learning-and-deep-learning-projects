# Name : Archana Kalburgi 
# CWID : 10469491
# Problem #3 Solution -> ANN 

remove(list=ls())

# Load the “IBM_Attrition_v3.csv”. 
ibm <-read.csv("/Users/archanakalburgi/Downloads/IBM_Attrition_v3.csv", na.strings = "?" , stringsAsFactors=TRUE)
# View(ibm)

# Convert all the variables to numeric variables 
ibm$Age = as.numeric(factor(ibm$Age))
ibm$JobSatisfaction = as.numeric(factor(ibm$JobSatisfaction))
ibm$MaritalStatus = as.numeric(factor(ibm$MaritalStatus))
ibm$MonthlyIncome = as.numeric(factor(ibm$MonthlyIncome))
ibm$YearsAtCompany = as.numeric(factor(ibm$YearsAtCompany))
ibm$Attrition = as.numeric(ibm$Attrition, levels = c('Yes','No'),labels = c(1,2))
# ibm$Attrition = as.numeric(factor(ibm$Attrition, levels = c('Yes','No'),labels = c(1,2)))
class(ibm$Attrition)

# a) Delete rows with missing values 
ibm_clean = na.omit(ibm)
# View(ibm_clean)

# b) Select 30% of the records as the test dataset and the remaining records as the training dataset
index<-sort(sample(nrow(ibm_clean),round(.30*nrow(ibm_clean))))
training<- ibm_clean[-index,]
test<- ibm_clean[index,]

# c)	Perform ANN, with 6 hidden nodes, for classification for the “attrition” column
library("neuralnet")
net_ibm = neuralnet(Attrition~Age+JobSatisfaction+MaritalStatus+MonthlyIncome+YearsAtCompany, training, hidden=6, threshold=0.01)
 # plot neural net
plot(net_ibm) 

# d)	Score the test dataset
ann <-compute(net_ibm , test[,0:5])
ann$net.result 

ann_cat<-ifelse(ann$net.result <1.5,1,2)
length(ann_cat)

table(Actual=test$Attrition,predition=ann_cat)

# e)	Measure the error rate. 
wrong<- (test$Attrition!=ann_cat)
error_rate<-sum(wrong)/length(wrong)
error_rate

remove(list=ls())
