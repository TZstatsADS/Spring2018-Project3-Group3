library(kernlab)
# Training and test data
tr_class <- read.csv("Feature_v3_TrainClass.csv", header = TRUE)
tr_data <- read.csv("Feature_v3_TrainData.csv", header = TRUE)

te_class <- read.csv("Feature_v3_TestClass.csv", header = TRUE)
te_data <- read.csv("Feature_v3_TestData.csv", header = TRUE)


train <- cbind(tr_class[,-1], tr_data[,-1])
colnames(train)[1] <- "class"

test <- cbind(te_class[,-1], te_data[,-1])
colnames(test)[1] <- "class"

# Tuning parameters

# svm radial kernel, tuning
svm_grid = expand.grid( C = c(2 ^ (-2:2)),
                        sigma = c(2 ^ (-3:3)))
svm_control = trainControl(method = "cv", number = 5,
                           returnResamp = "all", verbose = FALSE)
set.seed(42)
rad_svm_fit = train(class ~ ., data = train, method = "svmRadial",
                    trControl = svm_control, tuneGrid = svm_grid)
#rad_svm_fit
rad_svm_fit$bestTune

# model and time
old <- Sys.time()
train$class <- as.character(train$class)
test$class <- as.character(test$class)
rad_svm_fit = ksvm(class ~ ., data = train, kernel = 'rbfdot', C = 32, sigma = 2)
new1 <- Sys.time() - old
new1

# Accuracy function
accuracy = function(actual, predicted) {
  mean(actual == predicted)
}

# train accuracy
train_acc <- accuracy(actual = train$class,
                       predicted = predict(rad_svm_fit, train))
train_acc
# test accuracy
test_acc <- accuracy(actual = test$class,
                      predicted = predict(rad_svm_fit, test))
test_acc

