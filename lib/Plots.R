###################################
# Presentation Plots
###################################
# Plot of Log Loss

loss.data <- data.frame(x=c(1:5000),y=cv_model[["evaluation_log"]][, 4])
ggplot(data = loss.data)+
  geom_line(mapping = aes(x=loss.data[,1],y=loss.data[,2]))+
  geom_point(mapping = aes(x=min_logloss_index,y=min_logloss),color = "red")+
  geom_text(mapping = aes(x=min_logloss_index,y=min_logloss+0.05, label = 
                            paste("Optimal","(",round(min_logloss_index,2),",",round(min_logloss,2),")")))+
  labs(title="Log Loss of 10-fold Cross Validation")+
  theme(plot.title = element_text(size = 15, color = "black", face = "bold", vjust = 0.5, hjust = 0.5, angle = 0))+
  theme(axis.text.x = element_text(size = 10, color = "black", face = "bold", vjust = 0.5, hjust = 0.5, angle = 0))+
  theme(axis.text.y = element_text(size = 10, color = "black", face = "bold", vjust = 0.5, hjust = 0.5, angle = 0))+ 
  xlab("Nrounds") + 
  theme(axis.title.x = element_text(size = 12, color = "black", face = "bold", vjust = 0.5, hjust = 0.5, angle = 0))+
  ylab("Log Loss") + 
  theme(axis.title.y = element_text(size = 12, color = "black", face = "bold", vjust = 0.5, hjust = 0.5, angle = 90))

# Plot of Result
acc <- c(0.8624,0.8377,0.9103,0.8831,0.8913,0.8280,0.7950)
use.time <- c(313.56,61.21,440.56,190.51,721.95,146.64,5.64)
algo.name <- c("GBM","SVM(RBF)","XgBoost","Random Forest",
               "AdaBoosting","Logistic","Tree")
table.df <- data.frame(accuracy = acc,time = use.time,name = algo.name)
ggplot(data = table.df)+
  geom_point(mapping = aes(x=(time),y=accuracy), size = 4)+
  geom_text(mapping = aes(x=time,y=accuracy+0.01, label = name), size = 4)+
  geom_point(mapping = aes(x=(time[3]),y=accuracy[3]),color = "red",size = 5)+
  geom_text(mapping = aes(x=time[3],y=accuracy[3]+0.01, label = name[3]),color = "red", size = 4)+
  labs(title="Time vs. Accuracy(Feature: SIFT + ImageRGB)")+
  theme(plot.title = element_text(size = 15, color = "black", face = "bold", vjust = 0.5, hjust = 0.5, angle = 0))+
  theme(axis.text.x = element_text(size = 10, color = "black", face = "bold", vjust = 0.5, hjust = 0.5, angle = 0))+
  theme(axis.text.y = element_text(size = 10, color = "black", face = "bold", vjust = 0.5, hjust = 0.5, angle = 0))+ 
  xlab("Time (seconds)") + 
  theme(axis.title.x = element_text(size = 12, color = "black", face = "bold", vjust = 0.5, hjust = 0.5, angle = 0))+
  ylab("Accuracy Rate") + 
  theme(axis.title.y = element_text(size = 12, color = "black", face = "bold", vjust = 0.5, hjust = 0.5, angle = 90))
