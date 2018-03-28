SelectFeature <- function(SIFTname = "../lib/SIFT_test.csv", SubGroup = 12, N = 3000)
{
    NR <- SubGroup
    NG <- SubGroup
    NB <- SubGroup
    RSubgrp <- seq(0, 1.082, length.out=NR +1)
    gSubgrp <- seq(0, 1.082, length.out=NG +1)
    bSubgrp <- seq(0, 1.082, length.out=NB +1)
    #create feature matrix, recieve SIFT feature
    Feature <- read.csv(SIFTname, header = FALSE)
    Feature <- Feature[,-1]
    Feature <- Feature[1:N,]
    #N <- nrow(Feature)
    RGBfeature <- data.frame(matrix(0, nrow = N, ncol = NR*NG*NB))
    # file name
    ImageName <- function(x)
    {
      if (x < 10) return(paste("000", x, ".jpg", sep = ""))
      if (x < 100) return(paste("00", x, ".jpg", sep = ""))
      if (x < 1000) return(paste("0", x, ".jpg", sep = ""))
      return(paste(x, ".jpg", sep = ""))
    }
    for (k in 1:N)
    {
      mat <- readImage(paste("../lib/images/", ImageName(k), sep = ""))
      Interval <- data.frame(findInterval(mat[,,1], RSubgrp))
      Interval[,2] <- findInterval(mat[,,2], gSubgrp)
      Interval[,3] <- findInterval(mat[,,3], bSubgrp)
      #1,2,3 -> r,g,b, 4-> mix together
      Interval[,4] <- (Interval[,1] -1) * NG * NB + (Interval[,2] -1) * NB + (Interval[,3])
      Count <- table(Interval[,4])
      # calc frequency
      RGBfeature[k, as.numeric(names(Count))] <- as.numeric(Count) / ncol(mat) / nrow(mat)
      #cat(k,"\n")
    }
    return(cbind(Feature, RGBfeature))
    #write.csv(cbind(Feature, RGBfeature), "Feature_test.csv")
}
