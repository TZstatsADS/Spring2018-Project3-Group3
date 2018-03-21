train_rf = function(trainall){
  begin = Sys.time()
  rf <-randomForest(train_yall~.,data=trainall,
                    importance=TRUE,mtry=49,
                    ntree=200,keep.forest=TRUE,
                    nodesize=3)
  end= Sys.time()
  time = end - begin #Total Running Time
  return(list(fit = rf, time = time))
  
}

rfall <- train_rf(trainall)
rfpred <- predict(rfall$fit,test_xall)
rf01 <- as.numeric(rfpred == test_yall)
save(rfpred,file="rfpred.123.RData")
save(rf01,file="rf.01.RData")
mean(as.numeric(predict(rfall$fit,test_xall)==test_yall))
