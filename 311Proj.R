library("tiff")
library("cluster")
library("data.table")

x<- list.files(path ="jaffe")
df <- data.frame()

#load all collapsed data into a data frame
for(i in 1:length(x)){
 y<-(as.data.frame(readTIFF(paste("jaffe/",trim(x[i]),sep=""))))
 z<-unlist(y)
 z<-c(z, substring(trim(x[i]),3,5))
 df <- rbind(df,z)
}

head(df,1)
names <- seq.int(1, ncol(df), 1)
setnames(df,old = colnames(df),new=as.character(names))
colnames(df)

#choose k=7 as we know the number of groups
km <- kmeans(df,7)
clusplot(df,km$cluster, color = TRUE, shade= TRUE)

#run hclust for each linkage
cLink <- hclust(dist(df), method="complete")
sLink <- hclust(dist(df), method="single")
aLink <- hclust(dist(df), method="average")

#plot dendrogram
plot(cLink)
plot(sLink)
plot(aLink)

#seperate grouping based on dendrograms and the known categories
cCut <- cutree(cLink, k=7)
sCut <- cutree(sLink, k=7)
aCut <- cutree(aLink, k=7)

#plot groups based on dendrograms
plot(cCut)
plot(sCut)
plot(aCut)

#dislay first 6 rows and 4 columns
head(df[,1:5],n=6)
