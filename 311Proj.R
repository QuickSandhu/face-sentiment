library("tiff")
library("cluster")
library("randomForest")
library("graphics")

X<- as.data.frame(list.files(path ="jaffe"))
df <- data.frame()

#collapses a dataframe into a column
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

#load all collapsed data into a data frame
df<-as.data.frame(t(data.frame(apply(X,1, columnshift))))

#choose k=7 as we know the number of groups
km <- kmeans(df, 5)
clusplot(df,km$cluster, color = TRUE, shade= TRUE)

cLink <- hclust(dist(df), method="complete")
sLink <- hclust(dist(df), method="single")
aLink <- hclust(dist(df), method="average")

plot(cLink)
plot(sLink)
plot(aLink)

cCut <- cutree(cLink, k=7)
sCut <- cutree(sLink, k=7)
aCut <- cutree(aLink, k=7)

plot(cCut)
plot(sCut)
plot(aCut)

faceforest <-  randomForest(V1~., data=df[-1:-212,], importance=TRUE)


