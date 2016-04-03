library("tiff")
library("cluster")
library("h2o")

# Use convolution neural network to perform classification on the Japanese Face Database 
# image set.



x<- list.files(path ="jaffe")
df <- data.frame()

#load all collapsed data into a data frame
for(i in 1:length(x)){
  y<-readTIFF(paste("jaffe/",x[i],sep=""))
  if (length(dim(y)) > 2){
    y <- y[,,1]
  }
  
  z<-as.list(y)
  df <- rbind(df,z)
}


# simple neural network
library(neuralnet)
facenet <- neuralnet( formula = paste("y ~ ", paste() ), data = df, hidden = c(200,100), )