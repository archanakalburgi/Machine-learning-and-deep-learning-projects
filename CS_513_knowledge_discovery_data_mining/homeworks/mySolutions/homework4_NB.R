# Name : Archana Kalburgi
# CWID : 10469491
# Homework4 : Naive Bayes 

rm(list=ls())

install.packages('e1071', dependencies = TRUE) 
install.packages('caret', dependencies = TRUE) 

library(e1071)
library(class) 
library("caret")
library("lattice")
library("ggplot2")

csvfile <-file.choose()
breast_cancer <- read.csv(csvfile, na.strings = "?", colClasses=c("Class"="factor" ))
is.factor(breast_cancer$Class)



View(breast_cancer)
class(breast_cancer)

bc_clean <- na.omit(breast_cancer)

# setting labels :
# 2 for benign
# 4 for malignant 
bc_clean$Class <- factor(bc_clean$Class, levels = c(2,4), labels = c("Benign", "Malignant")) 

# extracting the feature variables 
bc_features = bc_clean[, 2:11] 
View(bc_features)

# splitting the data into test and train data 
# 70% training data and 30% test data  
Split <- sort (sample (nrow (bc_features), round (.70*nrow(bc_features))))

training <- bc_features[Split,]
test <- bc_features[-Split,]


# Applying naive bayes model to training dataset 
classifier <- naiveBayes(Class~ ., data = training)
classifier

# predicting on test data 
y_pred <- predict(classifier, newdata = test) 
y_pred 

# wrong<- ( test$Class!=Class)
# rate<-sum(wrong)/length(wrong)
# rate


# confusion matrix
cm <- table(test$Class, y_pred)
prop.table(cm) 
cm

# Evaluating the model 
# (TP+TN)/Total = 120+74/205 = 0.9463
confusionMatrix(cm) 
