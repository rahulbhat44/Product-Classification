library(nnet)
library(neuralnet)
library(NeuralNetTools)
library(ROCR)
library(pROC)
library(ggplot2)


train <- read.csv("~/Documents/Kaggle_Otto/Data/train.csv",header=TRUE,na.strings = "?")
train <- train[,-1]
test <- sample(1:nrow(train), floor(nrow(train)*0.3))

test2 <- train[test,]
train2 <- train[-test,]

#neural network to fit the model
fi.nnet <- nnet(target ~ ., data=train2, size = 8, rang = 0.3, 
              decay = 5e-4, maxit = 10, linear.output = FALSE,threshold=0.01)
#not useful 

#plot(scaled.train$target)
#print(fi.nnet)
#par(mar = numeric(4), family = 'serif')
#plotnet(fi.nnet, alpha = 0.5)

#predict on the test data
nnet.predict<-predict(fi.nnet,test2[1:93],type="class")

#creating Confusion Matrix and Overall Statistics
print(confusionMatrix(nnet.predict, test2$target))

#conf <- table(pred=predicted, true=test$target)
#summary(conf)

#ROC
#predictions <- as.numeric(predict(fi.nnet,test2[,-94] , type = 'raw'))
#proc <- print(multiclass.roc(test2$target, predicted))

# Output
preds = data.frame(1:nrow(nn.predict),nn.predict)
names(preds) = c('id', paste0('Class_',1:9))
write.table(preds, file = "~/Documents/Kaggle_Otto/submission_nnet.csv", col.names = TRUE, row.names = FALSE, sep = ",")
