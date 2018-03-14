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

  
  end_time_rf = Sys.time() # Model End time
  
  end_time_rf - start_time_rf
  
  
  
  rf_time = end_time_rf - start_time_rf #Total Running Time
  
  rf <-randomForest(x=train_x,y=factor(train_y),xtest=test_x,ytest=factor(test_y),
                    importance=TRUE,mtry=rf.fit,
                    ntree=200,keep.forest=TRUE,
                    nodesize=3)
  pred <- predict(rf, test_x, type="response")
  accurate <- mean(pred == test_y)
  return(list(fit = rf, time = rf_time,accurate = accurate))
  
}

