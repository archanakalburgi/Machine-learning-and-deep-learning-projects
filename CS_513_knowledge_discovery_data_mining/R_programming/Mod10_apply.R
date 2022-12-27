#################################################
#  Company    : Stevens 
#  Project    : R Bootcamp 
#  Purpose    : apply
#  First Name  : Khasha
#  Last Name  : Dehnad
#  Id			    : 12345
#  Date       :
#  Comments   :

#################################################
##   Step:
## 
##
##
##
##
##
##
##
##
##
##
##
######################

# lapply(): Loop over a list and evaluate a function on each element
# sapply(): Same as lapply but try to simplify the result
# tapply(): Apply a function over subsets of a vector

x <- list(a = 1:5, b=20:30 ,c=5:1)
lx<-lapply(x, mean)
is.list(lx)

mmnorm3 <-function(x)
{z<-((x-min(x))/(max(x)-min(x))) 
return(z)                              
}
mmnorm3
lmmnorm3=lapply(x,mmnorm3)

sx<-sapply(x, mean)
is.list(sx)



data(iris)
View(iris)
 
?mean()
?lappy()
 
lapply(iris[,-5], mean,na.ram=TRUE)
?sapply()
sapply(iris[,-5], mean)

?tapply()
tapply(iris[,1],iris[,5] , mean, na.rm = TRUE)
?colnames()
colnames(iris)<-c(NULL,NULL,NULL,NULL,NULL)
cols<-colnames(iris, do.NULL = FALSE, prefix = "col")
colnames(iris)<-cols
cols<-colnames(iris)
for(i in 1:4){
  stat_iris<-tapply(iris[,i],iris[,5] , mean, na.rm = TRUE)
  print(c("Mean by species for: ",cols[i]))
  
  print(stat_iris)
}
