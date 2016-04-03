library("tiff")
library("cluster")

x<- list.files(path ="jaffe")
df <- data.frame()

#load all collapsed data into a data frame
#load all collapsed data into a data frame
for(i in 1:length(x)){
  y<-readTIFF(paste("jaffe/",x[i],sep=""))
  if (length(dim(y)) > 2){
    y <- y[,,1]
  }
  z<-as.list(y)
  df <- rbind(df,z)
}

#choose k=7 as we know the number of groups
km <- kmeans(df,7)
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
