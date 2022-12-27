rm(list=ls())

#################################################
#  name : Archana Kalburgi  
#  CWID : 10469491
# Mid-term Solution #5
#################################################

rm(list=ls())
# name <- file.choose()
# covid <- read.csv(name)
covid <- read.csv("/Users/archanakalburgi/Downloads/KDD-midterm_Solutions/COVID19_v2.csv", na.strings = "?")

covid[,2:7][covid[,2:7]=="?"] <- NA

covid <- na.omit(covid[,2:7])
View(covid)

# discretize MonthAtHospital using cut function 
covid$MonthAtHospital <- cut(covid$MonthAtHospital, breaks = c(0,5,Inf),include.lowest = TRUE)
levels(covid$MonthAtHospital) <- c("less than 6 mnts", "6 or more mnts")

# discretize age using cut function 
covid$Age <- cut(covid$Age, breaks = c(0,34,50,Inf))
levels(covid$Age) <- c("less than 35", "35-50" , "51 and over")

View(covid)

# factorising all the columns of the dataset using factor function 
covid$Age <- factor(covid$Age)
covid$Exposure <- factor(covid$Exposure )
covid$Cases <- factor(covid$Cases) 
covid$MonthAtHospital <- factor(covid$MonthAtHospital)
covid$MaritalStatus <- factor(covid$MaritalStatus) 
covid$Infected <- factor(covid$Infected) 


# dividing dataset into test(30%) and train set
idx<-sort(sample(nrow(covid),as.integer(.70*nrow(covid))))
training<-covid[idx,]
test<-covid[-idx,]

# loading the libraries
library(e1071)
library(e1071)
library(class) 
library("caret")
library("lattice")
library("ggplot2")

# applying naiveBayes() 
nBayes <- naiveBayes(factor(Infected)~., data = training)
category_all<-predict(nBayes,test)

# calculating the accuracy if the model 
table(NBayes=category_all, Infected=test$Infected)
NB_wrong<-sum(category_all!=test$Infected )
NB_error_rate<-NB_wrong/length(category_all)
NB_error_rate

cm <- table(test$Infected, category_all)
prop.table(cm) 
cm

# printing out confusion matrix
confusionMatrix(cm) 
