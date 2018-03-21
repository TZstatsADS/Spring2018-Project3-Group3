library(adabag)
train_x <- read.csv("TrainData_SIFT.csv")
train_y <- read.csv("TrainClass_SIFT.csv")
test_x <- read.csv("TestData_SIFT.csv")
test_y <- read.csv("TestClass_SIFT.csv")
train_x <- as.data.frame(train_x)[,-1]
train_y <- factor(as.data.frame(train_y)[,-1])
test_x <- as.data.frame(test_x)[,-1]
test_y <- factor(as.data.frame(test_y)[,-1])
train <- cbind(train_y,train_x)

train_x2512 <- read.csv("TrainData_RGB+SIFT.csv")
train_y2512 <- read.csv("TrainClass_RGB+SIFT.csv")
test_x2512 <- read.csv("TestData_RGB+SIFT.csv")
test_y2512 <- read.csv("TestClass_RGB+SIFT.csv")
train_x2512 <- as.data.frame(train_x2512)[,-1]
train_y2512 <- factor(as.data.frame(train_y2512)[,-1])
test_x2512 <- as.data.frame(test_x2512)[,-1]
test_y2512 <- factor(as.data.frame(test_y2512)[,-1])
train2512 <- cbind(train_y2512,train_x2512)

train_x1695 <- read.csv("TrainData_OnlyRGB1728+SIFT.csv")
train_y1695 <- read.csv("TrainClass_OnlyRGB1728+SIFT.csv")
test_x1695 <- read.csv("TestData_OnlyRGB1728+SIFT.csv")
test_y1695 <- read.csv("TestClass_OnlyRGB1728+SIFT.csv")
train_x1695 <- as.data.frame(train_x1695)[,-1]
train_y1695 <- factor(as.data.frame(train_y1695)[,-1])
test_x1695 <- as.data.frame(test_x1695)[,-1]
test_y1695 <- factor(as.data.frame(test_y1695)[,-1])
train1695 <- cbind(train_y1695,train_x1695)

train_x3695 <- read.csv("TrainData_RGB1728+SIFT.csv")
train_y3695 <- read.csv("TrainClass_RGB1728+SIFT.csv")
test_x3695 <- read.csv("TestData_RGB1728+SIFT.csv")
test_y3695 <- read.csv("TestClass_RGB1728+SIFT.csv")
train_x3695 <- as.data.frame(train_x3695)[,-1]
train_y3695 <- factor(as.data.frame(train_y3695)[,-1])
test_x3695 <- as.data.frame(test_x3695)[,-1]
test_y3695 <- factor(as.data.frame(test_y3695)[,-1])
train3695 <- cbind(train_y3695,train_x3695)

train_xall <- read.csv("TrainData_ALL.csv")
train_yall <- read.csv("TrainClass_ALL.csv")
test_xall <- read.csv("TestData_ALL.csv")
test_yall <- read.csv("TestClass_ALL.csv")
train_xall <- as.data.frame(train_xall)[,-1]
train_yall <- as.data.frame(train_yall)[,-1]
test_xall <- as.data.frame(test_xall)[,-1]
test_yall <- as.data.frame(test_yall)[,-1]
train_yall <- factor(train_yall)
test_yall <- factor(test_yall)
trainall <- cbind(train_yall,train_xall)

train_xall <- read.csv("Feature_v1_TrainData.csv")
train_yall <- read.csv("Feature_v1_TrainClass.csv")
test_xall <- read.csv("Feature_v1_TestData.csv")
test_yall <- read.csv("Feature_v1_TestClass.csv")
train_xall <- as.data.frame(train_xall)[,-1]
train_yall <- as.data.frame(train_yall)[,-1]
test_xall <- as.data.frame(test_xall)[,-1]
test_yall <- as.data.frame(test_yall)[,-1]
train_yall <- factor(train_yall)
test_yall <- factor(test_yall)
trainall <- cbind(train_yall,train_xall)



####SIFT
time <- 
for (i in 1:5){
begin =Sys.time()
m <- i*5
ada <- boosting(train_y~.,data=train,boos=T,mfinal=m,coeflearn="Zhu")
end=Sys.time()
end-begin
mean(predict.boosting(ada,newdata=test_x)$class==test_y)
vet <- names(ada$importance)[head(order(ada$importance,decreasing=T))]
}

####RGB+SIFT2512
begin=Sys.time()
ada2512 <- boosting(train_y2512~.,data=train2512,boos=T,mfinal=30,coeflearn="Zhu")
end=Sys.time()
end-begin
mean(predict.boosting(ada2512,newdata=test_x2512)$class==test_y2512)
vet2512 <- names(ada2512$importance)[head(order(ada2512$importance,decreasing=T))]


####RGB+SIFT1695
begin=Sys.time()
ada1695 <- boosting(train_y1695~.,data=train1695,boos=T,mfinal=30,coeflearn="Zhu")
end=Sys.time()
end-begin
acc1695<- mean(predict.boosting(ada1695,newdata=test_x1695)$class==test_y1695)
vet1695 <- names(ada1695$importance)[head(order(ada1695$importance,decreasing=T))]

####RGB+SIFT3695
begin=Sys.time()
ada3695 <- boosting(train_y3695~.,data=train3695,boos=T,mfinal=30,coeflearn="Zhu")
end=Sys.time()
end-begin
acc3695 <- mean(predict.boosting(ada3695,newdata=test_x3695)$class==test_y3695)
vet3695 <- names(ada3695$importance)[head(order(ada3695$importance,decreasing=T))]

####RGB+LBP+SIFT


begin=Sys.time()
adaall <- boosting(train_yall~.,data=trainall,boos=T,mfinal=30,coeflearn="Zhu")
end=Sys.time()
end-begin
accall <- mean(predict.boosting(adaall,newdata=test_xall)$class==test_yall)
vetall <- names(adaall$importance)[head(order(adaall$importance,decreasing=T))]

save(adaall,file="adaboosting")
