#  Course          :  
#  First Name      : Khasha
#  Last Name       : Dehnad
#  Id              : 12345
#  purpose         :  

## Step 0 clean up!!!


rm(list=ls())

file<-file.choose()
dsn<- read.csv(file) # titanic 

install.packages('randomForest')
dev.off()

library(randomForest)
 
?randomForest()
?importance()
?tuneRF()
file<-file.choose()
dsn<- read.csv(file )


?read.csv()
set.seed(123)
dsn2<-na.omit(dsn)
set.seed(111)
?ifelse

dsn$Class = factor(dsn$Class)
dsn$Sex = factor(dsn$Sex)
dsn$Age = factor(dsn$Age)
dsn$Survived = factor(dsn$Survived) 
dsn2$Class = factor(dsn2$Class)
dsn2$Sex = factor(dsn2$Sex)
dsn2$Age = factor(dsn2$Age)
dsn2$Survived = factor(dsn2$Survived) 

index<-sort(sample(nrow(dsn2),round(.30*nrow(dsn2))))
training<-dsn[-index,]
test<-dsn[index,]
install.packages('randomForest')
library(randomForest)

fit <- randomForest( Survived~., data=training, importance=TRUE, ntree=1000) # few -> not good representative sample 
importance(fit)
varImpPlot(fit)
Prediction <- predict(fit, test)
table(actual=test[,4],Prediction)


wrong<- (test[,4]!=Prediction )
error_rate<-sum(wrong)/length(wrong)
error_rate 

# MeanDecreaseGini : what if we didn't have a particular column 
# age is less importan tthan the gender 
