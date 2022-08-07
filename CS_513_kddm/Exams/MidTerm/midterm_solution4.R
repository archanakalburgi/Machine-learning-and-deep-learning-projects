rm(list=ls())

#################################################
#  name : Archana Kalburgi  
#  CWID : 10469491
# Mid-term Solution #4
#################################################

rm(list=ls())
# name <- file.choose()
# covid <- read.csv(name)
covid <- read.csv("/Users/archanakalburgi/Downloads/KDD-midterm_Solutions/COVID19_v2.csv", na.strings = c("?"))
covid[,2:7][covid[,2:7]=="?"] <- NA
View(covid)

# remove the missing values
covid <- na.omit(covid[,2:7])

# loading libraries
library(kknn)
library(e1071)
library(e1071)
library(class) 
library("caret")
library("lattice")
library("ggplot2")


# Function for normalizing a column
mmnorm <-function(x,minx,maxx) {z<- ((x-minx)/(maxx-minx))
return(z) 
}

# normalizing all the columns of the dataset 
covid_normalized<-as.data.frame (         
  cbind( Age=mmnorm(covid[,1],min(covid[,1]),max(covid[,1])) 
         ,Exposure=mmnorm(covid[,2],min(covid[,2]),max(covid[,2]))
         ,MaritalStatus = as.character(covid[,3])
         ,Cases=mmnorm(covid[,4],min(covid[,4]),max(covid[,4]))
         ,MonthAtHospital=mmnorm(covid[,5],min(covid[,5]),max(covid[,5]))
        ,Infected=as.character(covid[,6])
))

# converting all the column values to a numeric values,
# since kknn function requires all the column values to be a numeric values and not character values
covid_normalized$Age <- as.numeric(covid_normalized$Age)
covid_normalized$Exposure <- as.numeric(covid_normalized$Exposure)
covid_normalized$Cases <- as.numeric(covid_normalized$Cases)
covid_normalized$MonthAtHospital <- as.numeric(covid_normalized$MonthAtHospital)
covid_normalized$MaritalStatus <- factor(covid_normalized$MaritalStatus, levels = c("Married", "Single","Divorced"))
covid_normalized$Infected <- factor(covid_normalized$Infected, levels = c("Yes", "No"))


# forming test and train dataset to predict infection rate (infected) for a random sample (30%) of the data (test dataset).
idx <- sample(nrow(covid),as.integer(.70*nrow(covid)))

test<-covid_normalized[idx,]
training <-covid_normalized[-idx,]

# predict infection rate (infected) for k=5 usinf kknn()
predict_k5<-kknn(formula = Infected ~ ., train = training, test[,-6],  k=5, kernel ="rectangular" )

fit <- fitted(predict_k5)
cm <- table(Actual=test$Infected,Fitted=fit)

# printing out the error rate
wrong<- ( test$Infected!=fit)
rate<-sum(wrong)/length(wrong)
rate

accuracy <- sum(diag(cm))/sum(cm)
accuracy

# creating confusion matrix to get the accuraccy of the model 
confusionMatrix(cm)
