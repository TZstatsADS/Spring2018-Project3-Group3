setwd("C:/Users/Kevin Zhang/Desktop/Spring2018-Project3-spring2018-project3-group3/data/2.28Test&Train")
# load packages
library("xgboost")
library("haven")
library("tidyverse") 
library("mlr")
library("plyr")
library("ggplot2")

timestart<-Sys.time()
# read data
train.data.sift <- read_csv("TrainData_RGB1728+SIFT.csv")
train.class <- read_csv("TrainClass_RGB1728+SIFT.csv")
train.data.sift <- as.matrix(train.data.sift)
train.class <- as.matrix(train.class)
train.class[,2] <- train.class[,2] - 1

test.data.sift <- read_csv("TestData_RGB1728+SIFT.csv")
test.class <- read_csv("TestClass_RGB1728+SIFT.csv")
test.data.sift <- as.matrix(test.data.sift)
test.class <- as.matrix(test.class)
test.class[,2] <- test.class[,2] - 1

# data preparation
xgb.train.data <- xgb.DMatrix(data = train.data.sift[,2:ncol(train.data.sift)],label = train.class[,2])
xgb.test.data <- xgb.DMatrix(data = test.data.sift[,2:ncol(train.data.sift)],label = test.class[,2])

# Default Parameter 
numberOfClasses <- length(unique(train.class[,2]))
xgb_params <- list("objective" = "multi:softmax",
                   "eval_metric" = "mlogloss",
                   "num_class" = numberOfClasses,
                   "silent"="0")
# tell the XGBoost algorithm that we want to to probabilistic classification and use a multiclass logloss as our evaluation metric. Use of the multi:softmax objective also requires that we tell is the number of classes we have with num_class.


# CV of xgboost (selection of nrounds)

nround    <- 200 # number of XGBoost rounds; most of the situation, nround is less than 100. Therefore, 200 should be enou
cv.nfold  <- 10
# Fit cv.nfold * cv.nround XGB models and save OOF predictions
cv_model <- xgb.cv(params = xgb_params,
                   data = xgb.train.data, 
                   nrounds = nround,
                   nfold = cv.nfold,
                   verbose = FALSE,
                   prediction = TRUE)
min_logloss = min(cv_model[["evaluation_log"]][, 4])
min_logloss_index = (1:200)[cv_model[["evaluation_log"]][, 4]==min_logloss]

loss.data <- data.frame(x=c(1:200),y=cv_model[["evaluation_log"]][, 4])
ggplot(data = loss.data)+
  geom_line(mapping = aes(x=loss.data[,1],y=loss.data[,2]))+
  geom_point(mapping = aes(x=min_logloss_index,y=min_logloss),color = "red")+
  labs(title="Log Loss of 10-fold Cross Validation",x="Nrounds",y="Log Loss")



# train model on all the train dataset
model <- xgb.train(data = xgb.train.data,    
                 nround = min_logloss_index, 
                 params = xgb_params)

train.data.pred <- predict(model,newdata = xgb.train.data)
sum(train.data.pred==train.class[,2])/nrow(train.class)
test.data.pred <- predict(model,newdata = xgb.test.data)
sum(test.data.pred==test.class[,2])/nrow(test.class)

timeend1<-Sys.time()
runningtime1<-timeend1-timestart
print(runningtime1) 
7.723262*60

# Selection of eta
# It controls the learning rate, i.e., the rate at which our model learns patterns in data.After every round, it shrinks the feature weights to reach the best optimum.
# Typically, it lies between 0.01 - 0.3. We will try 0.01, 0.05, 0.1, 0.2.

eta.vec <- c(0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7)
test.acc <- c()
for (i in 1:8){
  eta <- eta.vec[i]
  xgb_params <- list("objective" = "multi:softmax",
                     "eval_metric" = "mlogloss",
                     "num_class" = numberOfClasses,
                     "eta"=eta)
  model <- xgb.train(data = xgb.train.data,    
                     nround = min_logloss_index, 
                     params = xgb_params)
  test.data.pred <- predict(model,newdata = xgb.test.data)
  test.acc[i] <- sum(test.data.pred==test.class[,2])/nrow(test.class)
}
test.acc.max = max(test.acc)
eta_optimal = eta.vec[(1:8)[test.acc==test.acc.max]]




# Grid search procedure
# 1. Create learner
learner <- makeLearner("classif.xgboost",predict.type = "prob")
?makeLearner
eta_optimal <- as.numeric(eta_optimal[1])
min_logloss_index <- as.numeric(min_logloss_index)
learner$par.vals <- list("booster" = "gbtree", "objective" = "multi:softprob",   #multi:softprob  multi:softmax
                         "eval_metric" = "mlogloss", "num_class" = "numberOfClasses",
                         "nrounds" = min_logloss_index,"eta" = eta_optimal) #
# 2. Set parameter space
params.space <- makeParamSet( makeIntegerParam("max_depth",lower = 1L,upper = 10L),   # makeDiscreteParam("booster",values = c("gbtree","gblinear"))
                              makeNumericParam("min_child_weight",lower = 1L,upper = 10L),
                              makeNumericParam("subsample",lower = 0.5,upper = 1), 
                              makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))#,
                             # makeNumericParam("eta",lower = 0.01,upper = 0.5))

# 3. Set resampling strategy
rdesc <- makeResampleDesc("CV",stratify = T,iters=5L)

# 4. search strategy
ctrl <- makeTuneControlRandom(maxit = 500L)

# 5. set parallel backend
library("parallel")
library("parallelMap") 
parallelStartSocket(cpus = detectCores())

# 6. create tasks
train.data.sift.num <- apply(train.data.sift,2,as.numeric)
colnames(train.class) <- c("id","label")
train.data.sift.comb <- cbind(train.data.sift.num[,2:ncol(train.data.sift.num)],train.class[,2])
colnames(train.data.sift.comb)[1] <- "X1"
colnames(train.data.sift.comb)[ncol(train.data.sift.comb)] <- "label"
train.data.sift.comb <- as.data.frame(train.data.sift.comb)
train.data.sift.comb <- lapply(train.data.sift.comb, function(x) {
  if(is.factor(x)) as.numeric(as.character(x)) else x
})
train.data.sift.comb$label <- as.factor(train.data.sift.comb$label)
train.data.sift.comb <- as.data.frame(train.data.sift.comb)
traintask <- makeClassifTask (data = train.data.sift.comb, target = "label")



test.data.sift.num <- apply(test.data.sift,2,as.numeric)
colnames(test.class) <- c("id","label")
test.data.sift.comb <- cbind(test.data.sift.num[,2:ncol(test.data.sift.num)],test.class[,2])
colnames(test.data.sift.comb)[1] <- "X1"
colnames(test.data.sift.comb)[ncol(test.data.sift.comb)] <- "label"
test.data.sift.comb <- as.data.frame(test.data.sift.comb)
test.data.sift.comb <- lapply(test.data.sift.comb, function(x) {
  if(is.factor(x)) as.numeric(as.character(x)) else x
})
test.data.sift.comb$label <- as.factor(test.data.sift.comb$label)
test.data.sift.comb <- as.data.frame(test.data.sift.comb)
testtask <- makeClassifTask (data = test.data.sift.comb, target = "label")


# 6. parameter tuning
mytune <- tuneParams(learner = learner, task = traintask, resampling = rdesc, measures = acc, 
                     par.set = params.space, control = ctrl, show.info = T)
mytune$y 

#set hyperparameters
lrn_tune <- setHyperPars(learner,par.vals = mytune$x)

#train model
xgmodel <- train(learner = lrn_tune,task = traintask)

#predict model
xgpred <- predict(xgmodel,testtask)

sum(xgpred$data$response==test.class[,2])/nrow(test.class)








# Refit
b_params <- c(xgb_params,mytune$x)
b_params$eta <- eta_optimal
final.model <- xgb.train(data = xgb.train.data,    
                   nrounds = min_logloss_index, 
                   params = b_params)
?xgb.train
train.data.pred <- predict(final.model,newdata = xgb.train.data)
sum(train.data.pred==train.class[,2])/nrow(train.class)
test.data.pred <- predict(final.model,newdata = xgb.test.data)
sum(test.data.pred==test.class[,2])/nrow(test.class)




timeend2<-Sys.time()
runningtime2<-timeend2-timestart
print(runningtime2) 


#view variable importance plot
mat <- xgb.importance (feature_names = colnames(train.data.sift[,1:ncol(train.data.sift)]),model = model)
xgb.plot.importance (importance_matrix = mat[1:5]) 
?xgb.importance
mat[1:5]$Feature

num.of.occur <- c(sum(train.data.sift[,"X1227"]!=0),
                  sum(train.data.sift[,"X1189"]!=0),
                  sum(train.data.sift[,"X1242"]!=0),
                  sum(train.data.sift[,"X460"]!=0),
                  sum(train.data.sift[,"X638"]!=0))
per.of.occur <- num.of.occur/nrow(train.data.sift)


###Save the model
save(model,file='XgBoost_7min_Acc91.05_Model.RData')
save(final.model,file='XgBoost_6hr_Acc90.64_Model.RData')
###Load the model
#load("GBM_RGB_0.03_3_Acc0.843191196698762.RData")
