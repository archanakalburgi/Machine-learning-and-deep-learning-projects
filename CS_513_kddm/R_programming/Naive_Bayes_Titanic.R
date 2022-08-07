#  First Name       : Khasha
#  Last Name        : Dehnad   
#  Purpose          : Apply naive bayes to Titanic_row dataset 
#                   : see cr8_multi_rows_dsn program.
#  Creation date    : 

rm(list=ls())
installed.packages()
install.packages('e1071', dependencies = TRUE) 
# if this package depends on other packages r will dowload those other packages as well. 
 

library(e1071)
library(class) 
## main functions used in this program
?read.csv()
?naiveBayes()
?tabulate();  tabulate(c(2,3,3,5), nbins = 10);
?table()
?ftable()
?as.character()

# read in the Titanic dataset 

csvfile<-file.choose()
Titanic_rows<-  read.csv(csvfile)

View(Titanic_rows)
class(Titanic_rows)
?prop.table
prop.table # probability table



table(class=Titanic_rows$Class,Survival=Titanic_rows$Survived)
prop.table(table(Class=Titanic_rows$Class, Survived=Titanic_rows$Survived)) 
#divide all numbers by total number

## Naive Bayes classification using only one variable 
nBayes_class <- naiveBayes(Survived ~Class, data =Titanic_rows)
category_class<-predict(nBayes_class,Titanic_rows)
# applying result, bad way to do it (apply rest and train)
# classifier is nBayes_class




 

## Compare the prediction to actual
data_class<-cbind(Titanic_rows,category_class)
table(Class=Titanic_rows$Class,Survived=Titanic_rows$Survived)
table(Class=Titanic_rows$Class,NBayes=category_class)
table(NBayes=category_class,Survived=Titanic_rows$Survived)
?prop.table

prop.table(table(Class=Titanic_rows$Class,Survived=Titanic_rows$Survived))
## Naive Bayes classification using two variables 
nBayes_class_age <- naiveBayes(Survived ~Class+Age, data =Titanic_rows)
category_class_age<-predict(nBayes_class_age,Titanic_rows  )


## Compare the prediction to actual for two variables
table(Class=Titanic_rows$Class,Survived=Titanic_rows$Survived)
ftable(Class=Titanic_rows$Class,Titanic_rows$Age,Survived=Titanic_rows$Survived,NBayes=category_class_age,row.vars = 1:3)
ftable(Class=Titanic_rows$Class,Titanic_rows$Age,Survived=Titanic_rows$Survived,NBayes=category_class_age)
ftable(Class=Titanic_rows$Class,Titanic_rows$Age,Survived=Titanic_rows$Survived)


ftable(class=Titanic_rows$Class,Age=Titanic_rows$Age,Sex=Titanic_rows$Sex,
       Survival=Titanic_rows$Survived,row.vars = 1:3)
prop.table(
  ftable(class=Titanic_rows$Class,Age=Titanic_rows$Age,Sex=Titanic_rows$Sex,
         Survival=Titanic_rows$Survived,row.vars = 1:3)
)


nBayes_all <- naiveBayes(Survived ~., data =Titanic_rows)

## Naive Bayes classification using all variables 

category_all<-predict(nBayes_all,Titanic_rows  )
 

table(NBayes_all=category_all,Survived=Titanic_rows$Survived)
# 362+126 = 488
NB_wrong<-sum(category_all!=Titanic_rows$Survived)

NB_error_rate<-NB_wrong/length(category_all)
# 78% correct in the prediction 

# no normalization involved because it is categorical classification 
# categorical classification 
# data is conditional independent 


View(iris)
#split <- sample.split(iris, SplitRatio = 0.7) 
#train_cl <- subset(iris, split == "TRUE") 
#test_cl <- subset(iris, split == "FALSE") 

Split <- sort (sample (nrow (iris), round (.30*nrow(iris))))
training <- iris[-Split,]
test <- iris[Split,] 

train_scale <- scale(training[, 1:4]) 
test_scale <- scale(test[, 1:4]) 

set.seed(120)
classifier_cl <- naiveBayes(Species ~ ., data = training) 
classifier_cl

y_pred <- predict(classifier_cl, newdata = test) 


cm <- table(test$Species, y_pred)
cm

confusionMatrix(cm)
?confusionMatrix()

NB_wrong<-sum(y_pred!=iris$Species)

NB_error_rate<-NB_wrong/length(y_pred)

