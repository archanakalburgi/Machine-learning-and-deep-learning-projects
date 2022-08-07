# Name : Archana Kalburgi
# CWID : 10469491
# Homework7 : ANN 

remove(list=ls())

library("neuralnet")

filename <- file.choose()
bc <-  read.csv(filename , na.strings = "?" , stringsAsFactors=TRUE)


# omitting NA values
bc_clean <- na.omit(bc)

# factorize diagnosis column and convert it to numeric type
bc_clean$diagnosis = as.numeric(factor(bc_clean$diagnosis,levels = c('M','B'),labels = c(1,2)))
class(bc_clean$diagnosis)

# split train and test data 
index<-sort(sample(nrow(bc_clean),round(.30*nrow(bc_clean))))
training<- bc_clean[-index,]
test<- bc_clean[index,]

# apply neuralnet()
net_bc_clean<- neuralnet(diagnosis~radius_mean+texture_mean+perimeter_mean+area_mean
                              +smoothness_mean+compactness_mean+concavity_mean+
                                concave.points_mean+symmetry_mean+fractal_dimension_mean+
                                radius_se+texture_se+perimeter_se+area_se
                              +smoothness_se+compactness_se+concavity_se+
                                concave.points_se+symmetry_se+fractal_dimension_se+
                                radius_worst+texture_worst+perimeter_worst+area_worst
                              +smoothness_worst+compactness_worst+concavity_worst+
                                concave.points_worst+symmetry_worst+fractal_dimension_worst
                              ,training, hidden=5, threshold=0.01)

# plot the neural network
#plot(net_bc_clean)

# test should have only the input column
ann <-compute(net_bc_clean, test[,2:32])
ann$net.result

ann_cat<-ifelse(ann$net.result <1.5,1,2)
# length(ann_cat)
# length(test$diagnosis)

table(Actual=test$diagnosis,predition=ann_cat)

# performance
wrong<- (test$diagnosis!=ann_cat)
error_rate<-sum(wrong)/length(wrong)
error_rate
