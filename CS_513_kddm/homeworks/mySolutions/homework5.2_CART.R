############################################
# Name : Archana Kalburgi
# CWID : 10469491
# Assignment 5.2 
############################################

rm(list=ls())

library(rpart)
library(rpart.plot)  			# Enhanced tree plots
library(rattle)           # Fancy tree plot
library(RColorBrewer)     # colors needed for rattle



# load data into R
filename<-file.choose()
bc <-  read.csv(filename , na.strings = "?" , stringsAsFactors=TRUE)

View(bc) 
attach(bc) 
dev.off()

# remove all the records with missing value
bc <- na.omit(bc)

set.seed(111) # sets the random sample sequence 

# factorize all the columns 
bc$F1 = factor(bc$F1)
bc$F2 = factor(bc$F2)
bc$F3 = factor(bc$F3) 
bc$F4 = factor(bc$F4) 
bc$F5 = factor(bc$F5) 
bc$F6 = factor(bc$F6)
bc$F7 = factor(bc$F7) 
bc$F8 = factor(bc$F8) 
bc$F9 = factor(bc$F9) 
bc$Class = factor(bc$Class) 

# 25% test data , remaining as test data 
index<-sort(sample(nrow(bc),round(.25*nrow(bc))))
training<-bc[-index,]
test<-bc[index,]

# CART 
CART_class<-rpart(Class~.,data=training)
rpart.plot(CART_class)

CART_predict2 <- predict(CART_class,test, type="class") 
table(Actual=test[,11],CART=CART_predict2)
CART_predict<-predict(CART_class,test)  


CART_predict <-predict(CART_class,test)
str(CART_predict)

# calculating error rate 
CART_predict_cat<-ifelse(CART_predict[,2]<=.5,'2','4') 
table(Actual=test[,11],CART=CART_predict_cat)
CART_wrong<-sum(test[,11]!=CART_predict_cat) 
CART_error_rate<-CART_wrong/length(test[,11])
CART_error_rate

CART_predict2<-predict(CART_class,test, type="class")
CART_wrong2<-sum(test[,11]!=CART_predict2)
CART_error_rate2<-CART_wrong2/length(test[,11])
CART_error_rate2 

library(rpart.plot)
prp(CART_class)

fancyRpartPlot(CART_class)

