Data <- read.csv("SIFT_train.csv", header = FALSE)
head(Data)
Data <- Data[,-1]
names(Data) <- paste(paste("Var", 1:2000, sep = ""))
Lottery <- sample(1:100, 3000, replace = TRUE)
sum(Lottery <= 25)

Class <- read.csv("label_train.csv")

Train <- Data[which(Lottery > 25), ]
Test <- Data[which(Lottery <= 25), ]
TrainClass <- Class[which(Lottery > 25),3]
TestClass <- Class[which(Lottery <= 25),3]

write.csv(Train, "TrainData_SIFT.csv")
write.csv(Test, "TestData_SIFT.csv")
write.csv(TrainClass, "TrainClass_SIFT.csv")
write.csv(TestClass, "TestClass_SIFT.csv")


setwd("C:/Users/Kevin Zhang/Desktop/Spring2018-Project3-spring2018-project3-group3/data/2.28Test&Train")
# load packages
library("xgboost")
library("haven")
library("tidyverse") 

# read data
train.data.sift <- read_csv("TrainData_SIFT.csv")
train.class <- read_csv("TrainClass_SIFT.csv")
train.data.sift <- as.matrix(train.data.sift)
train.class <- as.matrix(train.class)
train.class[,2] <- train.class[,2] - 1

test.data.sift <- read_csv("TestData_SIFT.csv")
test.class <- read_csv("TestClass_SIFT.csv")
test.data.sift <- as.matrix(test.data.sift)
test.class <- as.matrix(test.class)
test.class[,2] <- test.class[,2] - 1

# data preparation
xgb.train.data <- xgb.DMatrix(data = train.data.sift[,2:2001],label = train.class[,2])
xgb.test.data <- xgb.DMatrix(data = test.data.sift[,2:2001],label = test.class[,2])
?xgb.DMatrix

numberOfClasses <- length(unique(train.class[,2]))
xgb_params <- list("objective" = "multi:softmax",
                   "eval_metric" = "mlogloss",
                   "num_class" = numberOfClasses)
# tell the XGBoost algorithm that we want to to probabilistic classification and use a multiclass logloss as our evaluation metric. Use of the multi:softprob objective also requires that we tell is the number of classes we have with num_class.


# cv of xgboost
nround    <- 1000 # number of XGBoost rounds
cv.nfold  <- 10
# Fit cv.nfold * cv.nround XGB models and save OOF predictions
cv_model <- xgb.cv(params = xgb_params,
                   data = xgb.train.data, 
                   nrounds = nround,
                   nfold = cv.nfold,
                   verbose = FALSE,
                   prediction = TRUE)
min_logloss = min(cv_model[["evaluation_log"]][, 4])
min_logloss_index = (1:1000)[cv_model[["evaluation_log"]][, 4]==min_logloss]

# train model
model <- xgb.train(data = xgb.train.data,    
                 nround = min_logloss_index, 
                 params = xgb_params)
test.data.pred <- predict(model,newdata = xgb.test.data)
sum(test.data.pred==test.class[,2])/nrow(test.class)
