library(ggplot2)
library(readr)
library(Rtsne)
library(tsne)

set.seed(123)
sample.rows <- 30000

train <- read.csv("~/Documents/Kaggle_Otto/Data/train.csv",header=TRUE,stringsAsFactors = F)
test <- read.csv("~/Documents/Kaggle_Otto/Data/test.csv",header=TRUE,stringsAsFactors = F)

# taking training data sample and removing id column
s.train <- train[sample(1:nrow(train), size = sample.rows),]
sample.train <- s.train[,c(-1, -95)]

#using Rtsne
Rtsne.out <- Rtsne(as.matrix(sample.train), pca = TRUE,
perplexity=50, theta=0.5, dims=2)

out <- as.data.frame(Rtsne.out$Y)
out$Class <- as.factor(sub("Class_", "", s.train[,95]))

Rtsne.plot <- ggplot(out, aes(x=V1, y=V2, color=Class)) +
geom_point(size=2.0) +
guides(colour = guide_legend(override.aes = list(size=8))) +
xlab("") + ylab("") +
ggtitle("t-SNE 2D Embedded Plot of Products Data") +
theme_light(base_size=15) +
theme(strip.background = element_blank(),
axis.ticks       = element_blank(),
axis.line        = element_blank(),
axis.text.x      = element_blank(),
axis.text.y      = element_blank(),
strip.text.x     = element_blank(),
panel.border     = element_blank())


plot(Rtsne.plot)

########------#########--------##########--------##########--------########----

train <- read.csv("~/Documents/Kaggle_Otto/Data/train.csv",header=TRUE,na.strings = "?")
train<- train[-1,]
sample.rows <- 1000

# taking training data sample and removing id column
s.train <- train[sample(1:nrow(train), size = sample.rows),]
sample.train <- s.train[,c(-1, -95)]

# K-Means Clustering with 20 clusters
fit<-kmeans(sample.train, 20)

# Cluster Plot against 1st 2 principal components
library(cluster)
clusplot(sample.train, fit$cluster, color=TRUE, shade=TRUE, labels=3, lines=0)

########--------######------#######--------######-------######-------#######

#train Visualization

library(ggplot2)
library(RColorBrewer)
library(gplots)
#source("http://bioconductor.org/biocLite.R")
#biocLite("Heatplus")  # annHeatmap or annHeatmap2
library(Heatplus)
library(vegan)
library(RColorBrewer)

set.seed(123)

train <- read.csv("~/Documents/Kaggle_Otto/Data/train.csv",header=TRUE,stringsAsFactors = F)
test <- read.csv("~/Documents/Kaggle_Otto/Data/test.csv",header=TRUE,stringsAsFactors = F)

train<-train[,-1]

new.train<-split(train[,-94],train$target)
sample.train<-lapply(new.train,colMeans)
sample.train<-matrix(unlist(sample.train),9,93,T)
distance<-dist(sample.train)
dist<-matrix(0,9,9)
k=1
for(i in 1:8){
    for(j in (i+1):9){
        dist[j,i]<-distance[k]
        dist[i,j]<-dist[j,i]
        k=k+1
    }
}
colnames(dist)<-c("Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9")
rownames(dist)<-colnames(dist)

#shows the distance from centers of each cluster
dist_heatmap<-heatmap(dist,Rowv=NA,Colv=NA,col=heat.colors(256), scale="column",margins=c(5,10), main="Correlation")




#ordinary pca
pca<-prcomp(~.,train[,-94])
pca_train<-data.frame(pca$x)
pca_group<-split(pca_train,train$target)
subindex<-sample(1:61878,2000)
sub_train<-train[subindex,]
sub_pca_train<-pca_train[subindex,]
sub_pca_group<-split(sub_pca_train,train$target[subindex])
plot(sub_pca_group$Class_1[,c(1,2)],col="red",xlim=c(-30,20),ylim=c(-30,20),main="2-D Projection plot of 2000 sample points.")
points(sub_pca_group$Class_2[,c(1,2)],pch=3,col="yellow")
points(sub_pca_group$Class_3[,c(1,2)],pch=2,col="blue")
points(sub_pca_group$Class_4[,c(1,2)],pch=20,col="grey")
points(sub_pca_group$Class_5[,c(1,2)],pch="o",col="black")
points(sub_pca_group$Class_6[,c(1,2)],pch="*",col="brown")
points(sub_pca_group$Class_7[,c(1,2)],pch=11,col="purple")
points(sub_pca_group$Class_8[,c(1,2)],pch=13,col="orange")
points(sub_pca_group$Class_9[,c(1,2)],pch=9,col="lavender")
legend(8,20,legend=colnames(dist),pch=c(1,3,2,20,111,42,11,13,9),col=c("yellow","blue","grey","black","brown","purple","orange","lavender"))

