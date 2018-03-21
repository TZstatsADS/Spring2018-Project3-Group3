#install package
library(neuralnet)
#Data preprocessing
traindata<- read.csv("data/Feature_v3_TrainData.csv")
trainclass<- read.csv("data/Feature_v3_TrainClass.csv")
traindata<- traindata[,-1]
traindata[is.na(traindata)] <- 0
testdata<- read.csv("data/Feature_v3_TestData.csv")
testclass<- read.csv("data/Feature_v3_TestClass.csv")
testdata<- testdata[,-1]
testdata[is.na(testdata)] <- 0
colnames(trainclass)<- c('x','class')
train_<- cbind(traindata, trainclass$class)
#train model
n <- names(train_)
f <- as.formula(paste("trainclass$class ~", paste(n[!n %in% "trainclass$class"], collapse = " + ")))
start<- Sys.time()
nn <- neuralnet(f,data=train_,hidden=5,
                linear.output = T)
end<- Sys.time()
#MSE calculation
pr.nn <- compute(nn,testdata)
#Accuracy and Time
results <- data.frame(actual = testclass$x, prediction = pr.nn$net.result)
results
roundedresults<-sapply(results,round,digits=0)
roundedresultsdf=data.frame(roundedresults)
table(roundedresultsdf$actual,roundedresultsdf$prediction)
test_accuray<-mean(roundedresultsdf$actual==roundedresultsdf$prediction)
time<- end-start
time
test_accuray

save(nn,file='NeuralNetwork.RData')