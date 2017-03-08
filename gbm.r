library(gbm)
library(caret)
library(ggplot2)
library(splines)
library(parallel)
require(caret)
require(corrplot)
require(stats)
require(knitr)
require(ggplot2)
require(caret) 
require(Metrics) 
require(corrplot)
require(dplyr)
require(data.table)
require(MLmetrics) # to evaluate multi-class logarithmic loss
require(stringr)
require(Matrix)
require(igraph) 


train <- read.csv("~/Documents/Kaggle_Otto/Data/train.csv",header=TRUE,stringsAsFactors = F)
test <- read.csv("~/Documents/Kaggle_Otto/Data/test.csv",header=TRUE,stringsAsFactors = F)

str(train)
str(test)

tran <- train[,-1]

set.seed(12345)

# Create sample train and test datasets
sample.train <- train[sample(nrow(train), 8000, replace=FALSE),]
sample.test <- train[sample(nrow(train), 3000, replace=FALSE),]

# Creating and fitting a model using the target field
gbm.fit <- gbm(target ~ ., data=sample.train, distribution = "multinomial",
               n.trees=10, shrinkage=0.01, interaction.depth=8, cv.folds=2)


# model testing
best.iter <- gbm.perf(gbm.fit)
print(best.iter)

summary(gbm.fit,
        cBars=length(gbm.fit$target),
        n.trees=gbm.fit$n.trees,
        plotit=TRUE, 
        order=TRUE,
        method=relative.influence,
        normalize=TRUE)

plot.gbm(gbm.fit, 1, best.iter)

preds <- predict(gbm.fit, sample.test, n.trees=best.iter, type="response")

plot(density(preds)) 

# Make prediction
preds = matrix(preds,9,length(preds)/9)
preds = t(preds)

gbm.predict <- as.data.frame(preds)
names(gbm.predict) <- c("Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9")

#to make same test data and train data
preds <- rep(NA,3000)

for (i in 1:nrow(sample.test)) {
  preds[i] <- colnames(gbm.predict)[(which.max(gbm.predict[i,]))]}


#build confusion matrix
confusion = as.data.frame(table(preds,sample.test$target))

# Create a confusion matrix of predictions vs actuals
table(preds,sample.test$target)
confusionMatrix(preds,sample.test$target)

# Determine the error rate for the model
gbm.fit$error <- 1-(sum(preds==sample.test$target)/length(sample.test$target))
gbm.fit$error

# Output
gbm.predict = data.frame(1:nrow(gbm.predict),gbm.predict)
names(gbm.predict) = c('id', paste0('Class_',1:9))
write.table(gbm.predict, file = "~/Documents/Kaggle_Otto/submission_gbm3.csv", col.names = TRUE, row.names = FALSE, sep = ",")
