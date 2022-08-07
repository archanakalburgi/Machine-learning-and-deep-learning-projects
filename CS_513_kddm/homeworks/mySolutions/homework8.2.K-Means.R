# Name : Archana Kalburgi
# CWID : 10469491
# Homework7 : k-means 

remove(list=ls())

library("neuralnet")

filename <- file.choose()
bc <- read.csv(filename , na.strings = "?" , stringsAsFactors=TRUE)

# omitting NA values
bc_clean <- na.omit(bc)

# factorize diagnosis column and convert it to numeric type
bc_clean$diagnosis = as.numeric(factor(bc_clean$diagnosis,levels = c('M','B'),labels = c(1,2)))

k_means<- kmeans(bc_clean[,2:31],2,nstart = 10)
k_means$cluster
k_means$centers
table(k_means$cluster,bc_clean[,2])
