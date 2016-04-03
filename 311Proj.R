library("tiff")
library("cluster")

X<- data.frame(list.files(path ="jaffe"))
df <- data.frame()

#collapses a dataframe into a column
columnshift <- function(x){
  y<-(as.data.frame(readTIFF(paste("jaffe/",x,sep=""))))
  if (length(dim(y)) > 2){
    y <- y[,,1]
  }
  z<-unlist(y)
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
