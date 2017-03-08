##Random Forest
library(ggplot2)
library(caret)
library(randomForest)

#load data
train <- read.csv("~/Documents/Kaggle_Otto/Data/train.csv",header=TRUE , na.strings = "?")
test <- read.csv("~/Documents/Kaggle_Otto/Data/test.csv",header=TRUE ,na.strings = "?")

train <- train[,-1]

# make target a factor
#train$target = as.factor(train$target)

fit.train = randomForest(target ~., data = train, ntree=10, do.trace=50, importance=TRUE)
fit.train
plot(fit.train)
#text(train)
#pairs(train)

varImpPlot(fit.train)

predict.rf = predict(fit.train, test, type = "prob") 

#output
# use the random forest model to create a prediction
submit <- data.frame(id = test$id, predict.rf)
write.table(submit, file = "~/Documents/Kaggle_Otto/submission_xg4.csv", col.names = TRUE, row.names = FALSE, sep = ",")
