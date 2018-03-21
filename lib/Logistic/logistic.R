# Install Package
library('caret')
library('nnet')
# Prepare Data
traindata1<- read.csv("TrainData_SIFT.csv")
traindata1<- traindata1[,-1]
trainclass1<- read.csv("TrainClass_SIFT.csv")
testdata1<- read.csv("TestData_SIFT.csv")
testdata1<- testdata1[,-1]
testclass1<- read.csv("TestClass_SIFT.csv")
colnames(trainclass1)<- c('x','class')
train_1<- cbind(traindata1, y=factor(trainclass1$class))
# Train Model Function(including cross validation)
multinom_train <- function(train_data){
  multinom_fit <- multinom(formula = y ~ .,
                           data=train_data, MaxNWts = 100000, maxit = 500)
  top_models = varImp(multinom_fit)
  top_models$variables = row.names(top_models)
  top_models = top_models[order(-top_models$Overall),]
  return(list(fit=multinom_fit, top=top_models))
}

# run it:
start<- Sys.time()
multinomfit_train1 = multinom_train(train_1)
end<- Sys.time()
# Test error function
multinom_test <- function(test_data, fit){
  multinom_pred = predict(fit, type="class", newdata=test_data)
  return(multinom_pred)
}
# Time and Accuracy
multinomtest_result1 = multinom_test(testdata1,multinomfit_train1$fit)
table(testclass1$x,multinomtest_result1)
testaccuracy<- mean(testclass1$x==multinomtest_result1)
testaccuracy
time<- end-start
time
