


#########################################################
##  Purpose: Create C4.5 classification tree
##  Developer: KD         
##
#########################################################

#########################################################
##  Step 0: Clear the environment and load the data
##           
##
#########################################################



rm(list=ls())

filename<-file.choose()
dsn<-  read.csv(filename ) 
dev.off

### remove all the records with missing value

?na.omit()
dsn2<-na.omit(dsn)
set.seed(111)
?ifelse

dsn$Class = factor(dsn$Class)
dsn$Sex = factor(dsn$Sex)
dsn$Age = factor(dsn$Age)
dsn$Survived = factor(dsn$Survived) 

index<-sort(sample(nrow(dsn),round(.30*nrow(dsn))))
training<-dsn[-index,]
test<-dsn[index,]


#install.packages("C50", repos="http://R-Forge.R-project.org")
#install.packages("C50")
library('C50')
# View(dsn)
?C5.0
# C50  classification 
library('C50')
# everything should be a factor column ****
# use "lapply" for the same -> mod10_apply.R  ****
C50_class <- C5.0( Survived~.,data=training ) # gives error as the columns are not factoirized 

summary(C50_class )
# dev.off()
plot(C50_class)
C50_predict<-predict( C50_class ,test , type="class" )
table(actual=test[,4],C50=C50_predict)
wrong<- (test[,4]!=C50_predict)
c50_rate<-sum(wrong)/length(test[,4])
c50_rate
