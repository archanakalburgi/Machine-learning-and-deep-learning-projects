# Name : Archana Kalburgi
# CWID : 10469491
# Homework 8.1 : hclust


# read the csv file
remove(list=ls())
filename <- file.choose()

breast_cancer <- read.csv(filename)
View(breast_cancer)

# omitting NA values
bc_clean<- na.omit(breast_cancer)

# factorize diagnosis column and convert it to numeric type
bc_clean$diagnosis = as.numeric(factor(bc_clean$diagnosis,levels = c('M','B'),labels = c(1,2)))

bc_distance <- dist(bc_clean[,2:31])
hclust_results <- hclust(bc_distance)
plot(hclust_results)
hcluster<-cutree(hclust_results,2)
table(hcluster,bc_clean[,2])
