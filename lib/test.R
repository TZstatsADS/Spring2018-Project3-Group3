######################################################
### Fit the classification model with testing data ###
######################################################

### Author: Mingming Liu
### Project 3 group3
### ADS Spring 2018

### Fit the classfication model with testing data

### Input: 
###  - the fitted classification model using training data
###  -  processed features from testing images 
### Output: training model specification

####GBM by Zhongxing Xue
gbm_test <- function(gbm_fit, TestData)
{
  Test.pred <- predict(object = gbm_fit, newdata = TestData, n.trees = gbm.perf(gbm_fit, plot.it = FALSE), type = "response")
  Test.pred <- matrix(Test.pred, ncol =3)
  Test.pred <- apply(Test.pred, 1, which.max)
  return(Test.pred)
}



####randomForest by Mingming Liu
rf_test <- function(fit_train,dat_test){
  pred <- predict(fit_model,newdata=dat_test, type="response")
  return(pred)
}


####svm by Yuhan Cha
svm_test<- function(fit_model, dat_test){
  
  pred <- predict(fit_model, 
                  testdata)
return(pred)
}

####tree model by Yuhan Cha
tree_test<- function(fit_model, dat_test){
  
  pred <- predict(fit_model, 
                  testdata, 
                  type = "class")
  return(pred)
}

####Adaboosting by Mingming Liu
adaboost_test<- function(fit_model, dat_test){
  predict <- predict.boosting(fit_model,newdata=dat_test)
  adapredict <- as.numeric(predict$class)
  return(adapredict)
}

####CNN by Keran Li

nn_test<- function(fit_model, testdata){
  
  pred<- round(compute(fit_model,testdata),0)
  return(pred)
}


####logistic by Keran Li
logistic_test<- function(fit_model,testdata){
  
  pred <- predict(fit_model, 
                  newdata=testdata, 
                  type = "class")
  return(pred)
}


####Xgboost by Junkai Zhang
xgboost_test<- function(fit_model, testdata){
  # fit_model should be xgb$fit and testdata has to be a matrix
  xgb.test.data <- xgb.DMatrix(data = testdata)
  pred <- predict(fit_model,newdata = xgb.test.data)
  return(pred)
}
#pred <- xgboost_test(xgb$fit,testdata) # predicted output of the xgboost is 0, 1 and 2.
#sum(pred==test.class[,2] - 1)/nrow(test.class)