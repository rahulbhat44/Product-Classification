#load dependencies
library(randomForest)

#load data

train <- read.csv("~/Documents/Kaggle_Otto/train.csv")
test <- read.csv("~/Documents/Kaggle_Otto/test.csv")


# make target a factor
train$target = as.numeric(as.factor(train$target))-1

#regression
rf = randomForest(target ~., data = train, ntree=100, do.trace=50, importance=TRUE)
rf
plot(rf)
#text(train)
#pairs(train)
#qplot(HomeTeam, AwayTeam, color=FTR, data=train)
rf.predict = predict(rf, test[,-1])

id<-test[,1]
submission<-cbind(id,predicted)
write.table(submission, file = "~/Documents/Kaggle_Otto/submission_rf3.csv", col.names = TRUE, row.names = FALSE, sep = ",")
