library(randomForest)
library(ggplot2)
library(adabag)
library(data.table)
library(dplyr)
library(randomForest)
library(parallel)
library(e1071)
library(caret)
library(stats)

train_x <- read.csv("TrainData_RGB+SIFT.csv")
train_y <- read.csv("TrainClass_RGB+SIFT.csv")
test_x <- read.csv("TestData_RGB+SIFT.csv")
test_y <- read.csv("TestClass_RGB+SIFT.csv")
train_x <- as.data.frame(train_x)[,-1]
train_y <- as.data.frame(train_y)[,-1]
test_x <- as.data.frame(test_x)[,-1]
test_y <- as.data.frame(test_y)[,-1]
rf <- train_rf(train_x,train_y,test_x,test_y)



train_rf = function(train_x,train_y,test_x,test_y){
  ###  dat_train: processed features from images also contains label
  
  rfGrid = c(floor(sqrt(ncol(train_x)) : floor(sqrt(ncol(train_x) * 1.50))))
  rf.fit <- 0
  acc <- rep(0,times=length(rfGrid))
  start_time_rf = Sys.time() # Model Start Time 

  for (i in 1:length(rfGrid)){
   x <- randomForest(x=train_x,y=factor(train_y),xtest=test_x,ytest=factor(test_y),
               importance=TRUE,mtry=i,
               ntree=200,
               nodesize=3)
   acc[i] <- mean(apply(x$test$votes,1, which.max) == test_y)
   print(acc[i])
   if (max(acc) == acc[i]){
   rf.fit <- rfGrid[i]}
   }
 # rf.fit = train(factor(Label)~., data = dat_train,method = "rf",  trControl = fitControl,ntree = 500, #number of trees to grow
                 
  #               tuneGrid = rfGrid) # Parameter Tuning
  
  end_time_rf = Sys.time() # Model End time
  
  end_time_rf - start_time_rf
  
  
  
  rf_time = end_time_rf - start_time_rf #Total Running Time
  
  return(list(fit = rf.fit, time = rf_time,accurate = max(acc)))
  
}



ntree <- train_rf(train_x,train_y,test_x,test_y)


#acc2 <- rep(0,times=7)
#while (i<=8){
#  tree <- i*50
#  y <- randomForest(x=train_x,y=factor(train_y),xtest=test_x,ytest=factor(test_y),
 #                  importance=TRUE,mtry=ntree$fit,
#                   ntree=tree,
#                   nodesize=3)
#  acc2[i] <- mean(apply(y$test$votes,1, which.max) == test_y)
#  print(acc2[i])

 # if (max(acc2) == acc2[i]){
#    rf.tree <- tree
#    }
# i = i+1
#    }

rf <-randomForest(x=train_x,y=factor(train_y),xtest=test_x,ytest=factor(test_y),
                  importance=TRUE,mtry=51,
                  ntree=200,keep.forest=TRUE,
                  nodesize=3)
system.time(pred <- predict(rf$fit, test_x, type="response"))
mean(pred == test_y)


rf.fit <- randomForest(x=train_x,y=factor(train_y),xtest=test_x,ytest=factor(test_y),
                  importance=TRUE,mtry=ntree$fit,
                  ntree=200,
                  nodesize=3)

