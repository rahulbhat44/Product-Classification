library(mxnet)    
library(ggplot2)  
library(Hmisc)    
library(dplyr)   
library(nnet) 
require(mlbench)

train <- read.csv("~/Documents/Kaggle_Otto/Data/train.csv",header=TRUE,stringsAsFactors = F)
test <- read.csv("~/Documents/Kaggle_Otto/Data/test.csv",header=TRUE,stringsAsFactors = F)


dim(train)
str(train)

str(train)
train.xg = train[,2:94]
train.xg = gsub('Class_','',train.xg)
y = as.matrix(as.numeric(train.xg$target)-1)

##Build machine learning model
train.xg = as.matrix(train.xg)
mode(train.xg) = 'numeric'
test.xg = test[,2:94]
test.xg = as.matrix(test.xg)

######------######-----####----######------######------########-------#######---

train <- train[sample(nrow(train)),]
train.xg = train[,ncol(train)]
train.xg = gsub('Class_','',train.xg)
y = as.integer(train.xg)-1 
x = rbind(train[,-ncol(train)],test)
x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))
train.ind = 1:length(y)
test.ind = (nrow(train)+1):nrow(x)

#train.x=train[,2:94]
#train.y.ind=which(class.ind(train$target)==1,arr.ind=T)
#train.y=train.y.ind[,2]-1
#train.x.eval=train[50001:nrow(train),2:94]
#train.y.eval=model.matrix(~ train$target[50001:nrow(train)])
#test.x=test[,2:94]

model <- mx.mlp(data = x[train.ind,], y, hidden_node=10, out_node=2, out_activation="softmax",
                num.round=20, array.batch.size=15, learning.rate=0.07, momentum=0.9, 
                eval.metric=mx.metric.accuracy)

preds=predict(model, x[test.ind,])

#summary(preds)
#pred.label = max.col(t(test.ind))-1
#table(pred.label, test.ind)



#graph.viz(model$symbol$as.json())

# check the distribution of the predictions
ggplot(data=data.frame(preds[1,]), aes(preds[1,], fill = cut)) +
  geom_histogram(binwidth = 0.02, fill = "red", alpha = 0.2)


# attach to data.frame
testprwid=data.frame(t(test.ind))
colnames(testprwid)=c("Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9")
testpr=data.frame(id=test$id,testprwid)
rownames(testpr) = NULL
write.table(output, file = "~/Documents/Kaggle_Otto/submission_mx.csv", col.names = TRUE, row.names = FALSE, sep = ",")

