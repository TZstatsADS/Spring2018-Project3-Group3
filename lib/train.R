#########################################################
### Train a classification model with training images ###
#########################################################

### Mingming Liu
### Project 3 Group 3
### ADS Spring 2018

### Train  GBM, randomforest,  using processed features from training images
### Input: 
###  -  processed features from images 
###  -  class labels for training images
### Output: training model specification and time used to train the model


####GBM by Zhongxing Xue

train_GBM <- function(traindata, shr = 0.03, dep = 3, ntree = 800) {
  ####this parameter is determined by cross vaildation
  TrainClass <- traindata[,1]
  TrainData <- traindata[,-1]
  timestart<-Sys.time()
  gbm_fit <- gbm.fit(x = TrainData, y = TrainClass,
                     distribution = "multinomial",
                     n.trees = ntree,
                     shrinkage = shr,
                     interaction.depth = dep, 
                     nTrain = 0.8 * length(TrainClass),
                     verbose = TRUE)
  timeend<-Sys.time()
  runningtime<-timeend-timestart
  return(list(fit = gbm_fit, time = runningtime))
}


####Random Forest by Mingming Liu
train_rf = function(trainall){
  begin = Sys.time()
  rf <-randomForest(class~.,data=trainall,
                    importance=TRUE,mtry=49,
                    ntree=200,keep.forest=TRUE,
                    nodesize=3)
 ##### the mtry and ntree are determined by cross vaildation
  end= Sys.time()
  time = end - begin #Total Running Time
  return(list(fit = rf, time = time))
  
  
####SVM by Yuhan Cha  
  train_svm = function(traindata){
    begin = Sys.time()
    svm <-ksvm(class~.,data=traindata,
               kernel = 'rbfdot', 
               C = 32, 
               sigma = 2)
    #####the parameter is determined by cross vaildation.
    end= Sys.time()
    time = end - begin #Total Running Time
    return(list(fit = svm, time = time))
  }
  
  
#### tree by Yuhan Cha  
  train_tree = function(traindata){
    begin = Sys.time()
    tree <- tree(as.factor(class) ~ ., data = traindata)
    end= Sys.time()
    time = end - begin #Total Running Time
    return(list(fit = tree, time = time))
  }
}


####Adaboost by Mingming Liu
train_adaboosting <- function(trainall){
  begin=Sys.time()
  adaall <- boosting(class~.,data=trainall,boos=T,mfinal=30,coeflearn="Zhu")
  #######the parameter is determined by cross vaildation
  end=Sys.time()
  boosttime = end-begin
  return(list(fit=adaall,time=boosttime))
}

####CNN by Keran Li
train_nn = function(traindata){
  begin = Sys.time()
  n <- names(traindata)
  f <- as.formula(paste("traindata$class ~", paste(n[!n %in% "traindata$class"], collapse = " + ")))
  model <- neuralnet(f,data=traindata,hidden=1,
                     linear.output = T)
  end= Sys.time()
  time = end - begin #Total Running Time
  return(list(fit = model, time = time))
}


####Logistic by Keran Li
train_logistic = function(traindata){
  begin = Sys.time()
  model=multinom_train(traindata)
  end= Sys.time()
  time = end - begin #Total Running Time
  return(list(fit = model, time = time))
}

####Xgboost by Junkai Zhang
train_xgboost <- function(traindata){
  # traindata has to be a matrix
  timestart <- Sys.time()
  # Data Preparation
  xgb.train.data <- xgb.DMatrix(data = traindata[,-1],label = traindata[,1] - 1)
  # Default Parameter 
  numberOfClasses <- length(unique(traindata[,1]))
  xgb_params <- list("objective" = "multi:softmax",
                     "eval_metric" = "mlogloss",
                     "num_class" = numberOfClasses,
                     "silent"="0")
  cv_model <- xgb.cv(params = xgb_params,
                     data = xgb.train.data, 
                     nrounds = 200,
                     nfold = 10,
                     verbose = FALSE,
                     prediction = TRUE)
  min_logloss = min(cv_model[["evaluation_log"]][, 4])     
  min_logloss_index = (1:200)[cv_model[["evaluation_log"]][, 4]==min_logloss]   
  xgb_fit <- xgb.train(data = xgb.train.data,    
                       nround = min_logloss_index, 
                       params = xgb_params)
  timeend <- Sys.time()
  runningtime <- timeend - timestart
  return(list(fit = xgb_fit, time = runningtime))
}
