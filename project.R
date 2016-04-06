#all of the following packages must be installed and included
install.packages("tiff")
install.packages("cluser")
install.packages("mclust")
install.packages("FNN")
install.packages("tree")
install.packages("plyr")

library("tiff")
library("cluster")
library("mclust")
library("FNN")
library("tree")
library("plyr")

#the data can be found at http://www.kasrl.org/jaffe_info.html
#the first link on that website is the download link
#in the zip file transfer the jaffe folder into the working directory
#or in the following path variable put the filepath to the jaffe folder

X<- data.frame(list.files(path ="jaffe"))
df <- data.frame()

#function to collapse all data and append sentiment rating
columnshift <- function(x){
  y<-(readTIFF(paste("jaffe/",x,sep="")))
  if (length(dim(y)) > 2){
    y <- y[,,1]
  }
  #reduce background noise by croping left/right margins
  y<-y[,-1:-45]
  y<-y[,-166:-211]
  
  #make Y a dataframe
  y <- as.data.frame(y)
  z <- unlist(y) # make y into a single list
  z <- append(z, c(substring(x,4,5)), 0) # append classification
  return(z)
}

#all elements are factors so unfactor makes values numeric
unfactor <- function(x){
  y <-as.numeric(as.character(x))
  return(y)
}

#load all collapsed data into a data frame
df <- as.data.frame(t(data.frame(apply(X,1, columnshift))))

#make all but sentiment classification numerics
df <- cbind.data.frame(df[,1], apply(df[,2:ncol(df)],2, unfactor)) 
df<-rename(df, c("df[, 1]"="V1")) #fix nameing inconsistency

#formula with V1 (sentiment) as response and the next 4999 as predictors
#decided to not use all 42k because it would most assureadly fail
#which this also does but it fails less severely
formula <- paste("V1~", paste(colnames(df[2:5000]), collapse=" + "))


#create test and training data
ind <- sample(1:nrow(df), 173)
train <- df[ind,]
test <- df[-ind,]


#choose k=10 the number of women
set.seed(12345)
km10 <- kmeans(df[,-1],10)
clusplot(df,km10$cluster, color = TRUE, shade= TRUE)
sil <- silhouette(km10$cluster,dist(df))
plot(sil,col = c("red", "green", "blue", "purple","yellow","paleturquoise","plum2", "blueviolet","darkcyan","darkmagenta"))

#choose k=7 the number of expression
set.seed(12345)
km7 <- kmeans(df[,-1],7)
clusplot(df,km7$cluster, color = TRUE, shade= TRUE)
sil <- silhouette(km7$cluster,dist(df))
plot(sil,col = c("red", "green", "blue", "purple","yellow","paleturquoise","plum2"))


#KNN classification on training/testing sets
knnface <- knn(train[,-1], test[-1], cl = train[,1])
summary(knnface)
plot(knnface)
table(knnface,test[,1])

#misclassification rate
#will only work if we classify at least one of all 7 sentiments
#which happens reliabley but it does have a chance to not occur
1-(sum(diag(table(knnface,test[,1]))))/nrow(test)

#heirarchical clustering
cLink <- hclust(dist(df), method="complete")
sLink <- hclust(dist(df), method="single")
aLink <- hclust(dist(df), method="average")

#plot dendrogram
plot(cLink)
plot(sLink)
plot(aLink)

#look at group structure from dendrograms
cCut <- cutree(cLink, k=7)
sCut <- cutree(sLink, k=7)
aCut <- cutree(aLink, k=7)

plot(cCut)
plot(sCut)
plot(aCut)

#mixture models
mod1<-Mclust(dist(df), G=7)
summary(mod1)
plot(mod1)

#Scaled and unscalde PCA of data
pca <- prcomp(df[,-1], scale. = TRUE)
summary(pca)

#Scree plot
plot(pca,type="lines")

#Kmeans attempt
pcakm<-kmeans(round(pca$rotation[,1:94],4),7, iter.max = 30)
clusplot(pca$rotation[,1:94],pcakm$cluster, color = TRUE, shade= TRUE)

plot(pcakm$iter)

#Hclust attempt
cLinkpca <- hclust(dist(pca$rotation[,1:94]), method="complete")

#modesl that fail:
#lm
#glm
#nueral network
#classification tree
#random forest
