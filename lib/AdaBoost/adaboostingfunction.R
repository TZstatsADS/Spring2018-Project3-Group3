train_adaboosting <- function(trainall){
begin=Sys.time()
adaall <- boosting(train_yall~.,data=trainall,boos=T,mfinal=30,coeflearn="Zhu")
end=Sys.time()
boosttime = end-begin
return(list(fit=adaall,time=boosttime))
}


train_xall <- read.csv("Feature_v3_TrainData.csv")
train_yall <- read.csv("Feature_v3_TrainClass.csv")
test_xall <- read.csv("Feature_v3_TestData.csv")
test_yall <- read.csv("Feature_v3_TestClass.csv")
train_xall <- as.data.frame(train_xall)[,-1]
train_yall <- as.data.frame(train_yall)[,-1]
test_xall <- as.data.frame(test_xall)[,-1]
test_yall <- as.data.frame(test_yall)[,-1]
train_yall <- factor(train_yall)
test_yall <- factor(test_yall)
trainall <- cbind(train_yall,train_xall)
adaall <- train_adaboosting(trainall)

predict <- predict.boosting(adaall$fit,newdata=test_xall)
adapredict <- predict$class
save(adapredict,file="adapredict.test.result.RData")
result <- as.numeric(predict.boosting(adaall$fit,newdata=test_xall)$class==test_yall)
save(result,file="adaboosting.test.result.RData")
mean(result)