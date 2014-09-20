setwd("C:/Users/Chuan/My Study/CourseRA/JHU8 - Practical Machine Learning/5 My Projects")

rm(list=ls())
cat("\014")


#### Load training data set and testing data set ####
data.raw <- read.csv("pml-training.csv")
test.raw <- read.csv("pml-testing.csv")

#### preprocess training data ####
data <- data.raw
data[data==""] <- NA
data$cvtd_timestamp <- as.character(data$cvtd_timestamp)
data$cvtd_timestamp <- strptime(data$cvtd_timestamp, "%d/%m/%Y %H:%M")

data.num <- data.frame(data[,3:4],data[,7:(dim(data)[2]-1)])
for(i in 1:dim(data.num)[2])
{
    if(!is.numeric(data.num[1,i]))
    {
        data.num[,i] <- as.numeric(as.character(data.num[,i]))
    }
}
data.num <- data.num[,!is.na(colSums(data.num))]
data.no.cna <- data.frame(data[,1:2],data.num[,1:2],
                          data[,5:6],data.num[,3:dim(data.num)[2]],
                          data[,dim(data)[2]])
names(data.no.cna)[c(1:2,dim(data.no.cna)[2])] <- 
    names(data)[c(1:2,dim(data)[2])]

#### preprocess testing data set ####
test <- test.raw
test[test==""] <- NA
test$cvtd_timestamp <- as.character(test$cvtd_timestamp)
test$cvtd_timestamp <- strptime(test$cvtd_timestamp, "%d/%m/%Y %H:%M")

test.num <- data.frame(test[,3:4],test[,7:(dim(test)[2]-1)])
for(j in 1:dim(test.num)[2])
{
    if(!is.numeric(test.num[1,j]))
    {
        test.num[,j] <- as.numeric(as.character(test.num[,j]))
    }
}
test.num <- test.num[,!is.na(colSums(test.num))]
test.no.cna <- data.frame(test[,1:2],test.num[,1:2],
                          test[,5:6],test.num[,3:dim(test.num)[2]],
                          test[,dim(test)[2]])
names(test.no.cna)[c(1:2,dim(test.no.cna)[2])] <- 
    names(test)[c(1:2,dim(test)[2])]

#### Preparing for random forest training ####
library(caret)
features.train <- data.frame(data.no.cna[,c(2:4,6:(dim(data.no.cna)[2]))])
set.seed(5425)
inTrain <- createDataPartition(features.train$user_name,p=3/4,list=FALSE)
data.train <- features.train[inTrain,]
data.cv <- features.train[-inTrain,]

#### Train random forest model with 20 trees ####
library(randomForest)
set.seed(2609)
modFit <- randomForest(classe ~ ., data=data.train, ntree=1)

#### Check accuracy on the cross validation data set ####
pred   <- predict(modFit, newdata=data.cv)
confusionMatrix(table(pred,data.cv$classe))
# Confusion Matrix and Statistics
# 
# 
# pred    A    B    C    D    E
# A 1428    3    0    0    0
# B    0  936    1    0    0
# C    0    3  843    1    0
# D    0    0    0  801    2
# E    0    0    0    0  886
# 
# Overall Statistics
# 
# Accuracy : 0.998          
# 95% CI : (0.9963, 0.999)
# No Information Rate : 0.2912         
# P-Value [Acc > NIR] : < 2.2e-16      
# 
# Kappa : 0.9974         
# Mcnemar's Test P-Value : NA             
# 
# Statistics by Class:
# 
#                      Class: A Class: B Class: C Class: D Class: E
# Sensitivity            1.0000   0.9936   0.9988   0.9988   0.9977
# Specificity            0.9991   0.9997   0.9990   0.9995   1.0000
# Pos Pred Value         0.9979   0.9989   0.9953   0.9975   1.0000
# Neg Pred Value         1.0000   0.9985   0.9998   0.9998   0.9995
# Prevalence             0.2912   0.1921   0.1721   0.1635   0.1811
# Detection Rate         0.2912   0.1909   0.1719   0.1633   0.1807
# Detection Prevalence   0.2918   0.1911   0.1727   0.1637   0.1807
# Balanced Accuracy      0.9996   0.9967   0.9989   0.9991   0.9989

#### Make predictions on the test set ####
features.test <- data.frame(test.no.cna[,c(2:4,6:(dim(test.no.cna)[2]-1))])
features.test$classe <- tail(data.cv$classe,n=20)
for(i in 1:ncol(features.test))
{
    class(features.test[,i]) <- class(data.train[,i])
}
names(features.test) <- names(data.train)
x <- rbind(data.cv[100,],features.test)
features.test <- x[2:21,]

answers <- predict(modFit, newdata=features.test)

#### Output the predictions for submission ####
pml_write_files = function(x)
{
    n = length(x)
    for(i in 1:n)
    {
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

pml_write_files(answers)

## The predictions on the test set are all correct! In other words, its out-of-sample forecast accuracy is 100%, and the predictions are:
# 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
# B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
