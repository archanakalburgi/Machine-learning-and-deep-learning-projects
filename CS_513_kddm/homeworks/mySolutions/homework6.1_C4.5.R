############################################
# Name : Archana Kalburgi
# CWID : 10469491
# Assignment 6.1 
############################################

rm(list=ls())

# loading data into R
filename <- file.choose()
bc <- read.csv(filename) 

# remove all the records with missing value
bc_clean <- na.omit(bc)
View(bc_clean)

set.seed(111)

# factorizing all the columns 
bc_clean$Sample = factor(bc_clean$Sample)
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

# dividing data into test and train data
index<-sort(sample(nrow(bc_clean),round(.30*nrow(bc_clean))))
training<-bc_clean[-index,]
test<-bc_clean[index,]

# load C50 libraray into R
library('C50')


# C5.0  classification 
C50_class <- C5.0(Class~.,data=training)  

summary(C50_class )
plot(C50_class)
C50_predict<-predict( C50_class ,test , type="class" )

# calculate error rate 
table(actual=test[,11],C50=C50_predict)
wrong<- (test[,11]!=C50_predict)
c50_rate<-sum(wrong)/length(test[,11])
c50_rate
