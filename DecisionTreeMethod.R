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

#### Preparing for decision tree training ####
library(caret)
features.train <- data.frame(data.no.cna[,c(2:4,6:(dim(data.no.cna)[2]))])
set.seed(5425)
inTrain <- createDataPartition(features.train$user_name,p=3/4,list=FALSE)
data.train <- features.train[inTrain,]
data.cv <- features.train[-inTrain,]

#### Train decision tree model ####
library(rpart)
modFit <- rpart(classe ~ ., data=data.train)
fancyRpartPlot(modFit)

#### Check accuracy on the cross validation data set ####
pred   <- predict(modFit, newdata=data.cv, type="class")
confusionMatrix(table(pred,data.cv$classe))
# Confusion Matrix and Statistics
# 
# 
# pred    A    B    C    D    E
# A 1277   20    1    0    9
# B   92  788   60   52   17
# C    1   63  763   42    3
# D   21   71   20  666   44
# E   37    0    0   42  815
# 
# Overall Statistics
# 
# Accuracy : 0.8787          
# 95% CI : (0.8692, 0.8877)
# No Information Rate : 0.2912          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.8468          
# Mcnemar's Test P-Value : < 2.2e-16       
# 
# Statistics by Class:
# 
#                      Class: A Class: B Class: C Class: D Class: E
# Sensitivity            0.8943   0.8365   0.9040   0.8304   0.9178
# Specificity            0.9914   0.9442   0.9732   0.9620   0.9803
# Pos Pred Value         0.9770   0.7810   0.8750   0.8102   0.9116
# Neg Pred Value         0.9580   0.9605   0.9799   0.9667   0.9818
# Prevalence             0.2912   0.1921   0.1721   0.1635   0.1811
# Detection Rate         0.2604   0.1607   0.1556   0.1358   0.1662
# Detection Prevalence   0.2665   0.2058   0.1778   0.1676   0.1823
# Balanced Accuracy      0.9428   0.8904   0.9386   0.8962   0.9491

#### Make predictions on the test set ####
features.test <- data.frame(test.no.cna[,c(2:4,6:(dim(test.no.cna)[2]-1))])
features.test$classe <- tail(data.cv$classe,n=20)
for(i in 1:ncol(features.test))
{
    class(features.test[,i]) <- class(data.train[,i])
}
# names(features.test) <- names(data.train)
# x <- rbind(data.cv[100,],features.test)
# features.test <- x[2:21,]

answers <- predict(modFit, newdata=features.test, type="class")
# 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
# B  A  C  A  A  E  D  C  A  A  D  C  B  A  E  E  A  B  B  B 

#### True Values ####
# 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
# B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 

## Comparing the prediction and the true value, we can see that the out-of-sample forecast accuracy of the decision tree method is: 17/20 = 85%, which is slightly lower than its in-sample forecast accuracy (87.87%)
