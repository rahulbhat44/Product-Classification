# Boosting

library(gbm)
set.seed(2)

train <- read.csv("~/Documents/Kaggle_Otto/Data/train.csv",header=TRUE,stringsAsFactors = F)
test <- read.csv("~/Documents/Kaggle_Otto/Data/test.csv",header=TRUE,stringsAsFactors = F)


set.seed(17)

# Creating and fitting a model using the target field
gbm.fit <- gbm(target ~ ., data=train, distribution="multinomial", n.trees=10, 
            shrinkage=0.01, interaction.depth=10, cv.folds=2)

# Testing the boosting model on the sample test data which we have created
trees <- gbm.perf(gbm.fit)
gbm.predict <- predict(gbm.fit, test, n.trees=trees, type="response")
pred = matrix(gbm.predict,9,length(gbm.predict)/9)
pred = t(pred)

# Output
pred = data.frame(1:nrow(pred),pred)
names(pred) = c('id', paste0('Class_',1:9))
write.table(pred, file = "~/Documents/Kaggle_Otto/submission_xg4.csv", col.names = TRUE, row.names = FALSE, sep = ",")

