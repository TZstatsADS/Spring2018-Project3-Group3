library(tree)
# Training and test data
tr_class <- read.csv("Feature_v3_TrainClass.csv", header = TRUE)
tr_data <- read.csv("Feature_v3_TrainData.csv", header = TRUE)

te_class <- read.csv("Feature_v3_TestClass.csv", header = TRUE)
te_data <- read.csv("Feature_v3_TestData.csv", header = TRUE)


train <- cbind(tr_class[,-1], tr_data[,-1])
colnames(train)[1] <- "class"

test <- cbind(te_class[,-1], te_data[,-1])
colnames(test)[1] <- "class"

# model and time
old <- Sys.time()
tree = tree(as.factor(class) ~ ., data = train)
new1 <- Sys.time() - old
new1


tree_prune = prune.misclass(tree1, best = 9)
prune_trt_pred = predict(tree1, train, type = "class")
prune_tst_pred = predict(tree1, test, type = "class")
table(predicted = prune_tst_pred, actual = test$class)

# test accuracy
(tree_tst_acc = accuracy(predicted = prune_tst_pred, actual = test$class))


