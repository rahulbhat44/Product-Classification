#XGBoost


require(caret)
require(corrplot)
require(xgboost)
require(stats)
require(knitr)
require(ggplot2)
require(caret) #for dummyVars
require(Metrics) #calculate errors
require(corrplot)
require(dplyr)
require(data.table)
require(MLmetrics) # to evaluate multi-class logarithmic loss
require(stringr)
require(Matrix)
require(igraph) 


train <- read.csv("~/Documents/Kaggle_Otto/Data/train.csv",header = TRUE,sep=",")
test <- read.csv("~/Documents/Kaggle_Otto/Data/test.csv",header = TRUE,sep=",")

str(train)

y <- as.numeric(as.factor(train[["target"]])) - 1
X <- train[, -c(1, ncol(train))]
dim(X)

X_test <- test[, -c(1)]
dim(X_test)
All <- bind_rows(X, X_test)

# Scaling
All <- All %>%
  dplyr::mutate_each(funs(scale))
X <- data.matrix(All[1:nrow(X),])
X_test <- data.matrix(All[(nrow(X)+1):nrow(All),])

# Extract validation dataset
set.seed(123)
cv_list <- createFolds(y, k = 5, list = TRUE, returnTrain = FALSE)
X_train <- X[-cv_list[[1]], ]
dim(X_train)

X_valid <- X[cv_list[[1]], ]
dim(X_valid)

y_train <- y[-cv_list[[1]]]
y_valid <- y[cv_list[[1]]]


#MLogLoss function sometime works and sometimes not
mx.metric.mlogloss <- mx.metric.custom("mlogloss", function(label, pred){
  label_mat <- data.frame(i = 1:length(label),
                          j = label + 1,
                          x = rep(1, length(label)))
  label_mat <- sparseMatrix(i = label_mat$i,
                            j = label_mat$j,
                            x = label_mat$x)
  label_mat <- as.matrix(label_mat)
  return(MultiLogLoss(label_mat, t(pred)))
})


# Function to calculate Log Loss
LogLoss <- function(actual, predicted, eps=0.00001) {
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  -1/length(actual)*(sum(actual*log(predicted)+(1-actual)*log(1-predicted)))
}




# Set necessary parameters
param <- list("objective" = "multi:softprob",   # multiclass classification 
              "eval_metric" = "mlogloss",         # evaluation metric 
              "num_class" = 9,                  # number of classes 
              "nthread" = 8,                    # number of threads
              "bst:eta" = 0.2,                   # step size shrinkage 
              "bst:max_depth" = 8,             # maximum depth of tree 
              "gamma" = 1, 
              "subsample" = 0.85,         # part of data instances to grow tree 
              "colsample_bytree" = 0.85,  # subsample ratio of columns when constructing each tree 
              "min_child_weight" = 12 )   # minimum sum of instance weight needed in a child 

##Cross Validation
set.seed(1)
nround.cv = 10   #k-fold validaton, with timing
xgbst.cv <- xgb.cv(param=param, data=X_valid, label=y_valid,
                   nfold = 3, nrounds=nround.cv,  prediction=TRUE, 
                   verbose=FALSE, missing = NaN)

#head(xgbst.cv)
#summary(xgbst.cv)
#str(xgbst.cv)
#nround=which(xgbst.cv$test.merror.mean==min(xgbst.cv$test.merror.mean))
#min.merror = which.min(xgbst.cv$test.merror.mean) 
#min.merror

# check best iteration
#which(xgbst.cv$test.mlogloss.mean == min(xgbst.cv$test.mlogloss.mean))

##Model Training
bst <- xgboost(param=param, data=X_valid, label=y_valid, 
               nrounds=nround.cv)

#predict on test data
preds = predict(bst,X_valid)
summary(preds)

# Make prediction
preds = matrix(preds,9,length(preds)/9)
preds = t(preds)


y_valid_mat <- data.frame(i = 1:length(y_valid),
                          j = y_valid + 1,
                          x = rep(1, length(y_valid)))
y_valid_mat <- sparseMatrix(i = y_valid_mat$i,
                            j = y_valid_mat$j,
                            x = y_valid_mat$x)
y_valid_mat <- as.matrix(y_valid_mat)


# Evaluate multi-class logarithmic loss
print(MultiLogLoss(y_true=y_valid_mat, y_pred=preds))

#Evaluate logarithmic loss
print(LogLoss(actual=y_valid_mat, predicted=preds))

#Feature importance
model <- xgb.dump(bst, with_stats = T)
model[1:10]

# Get the feature real names
names <- dimnames(X_valid)[[2]]

# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model = bst)
importance_matrix

# Nice graph
xgb.plot.importance(importance_matrix[1:10,])

xgb.plot.tree(feature_names = names, model = bst, n_first_tree = 2)

# plot
importance_matrix = xgb.importance(feature_names = names, model = bst)
head(importance_matrix)
gp = xgb.plot.importance(importance_matrix)
print(gp) 

# Output
preds = data.frame(1:nrow(preds),preds)
names(preds) = c('id', paste0('Class_',1:9))
write.table(preds, file = "~/Documents/Kaggle_Otto/submission_xg45.csv", col.names = TRUE, row.names = FALSE, sep = ",")
