# Name : Archana Kalburgi 
# CWID : 10469491
# Problem 5 
# Columns considered : Returns_pct1 to Returns_pct6, STATES(STATES=NY)


remove(list=ls())
# file <-file.choose()
zip_codes <-  read.csv("/Users/archanakalburgi/Downloads/NYNJ_zipcode.csv")
# View(zip_codes)

table(zip_codes$STATE) 

# Normalize all columns
nmnorm <-function(x)
{z<-((x-min(x))/(max(x)-min(x)))
return(z)                              
}

myvector<-1:20
nmnorm(myvector)

zips_normal = data.frame(STATEFIPS = as.character(zip_codes$STATEFIPS)
                  , STATE =   as.factor(zip_codes$STATE)
                  , zipcodes = nmnorm(zip_codes$zipcode)
                  , Returns_pct1 = nmnorm(zip_codes$Returns_pct1)
                  , Returns_pct2 = nmnorm(zip_codes$Returns_pct2)
                  , Returns_pct3 = nmnorm(zip_codes$Returns_pct3)
                  , Returns_pct4 = nmnorm(zip_codes$Returns_pct4)
                  , Returns_pct5 = nmnorm(zip_codes$Returns_pct5)
                  , Returns_pct6 = nmnorm(zip_codes$Returns_pct6)
                  , total = nmnorm(zip_codes$Returns_pct6)) 


# Remove rows with STATE = NJ to consider rows only with STATE = NY 
zips = zips_normal[!zips_normal$STATE == "NJ",]


# Hierarchical clustering (centroid) -> 4 clusters 
zips_dist<-dist(zips[,c(4:9)])
hclust_results<-hclust(zips_dist) 
plot(hclust_results)

  # cut tree 
hclust2 <- cutree(hclust_results, 4)
plot(hclust2)

table(Cluster = hclust2, Actual = zips[,2])

dev.off()

# K-means, population (centroid) -> 4 clusters 
kmean_clust <- kmeans(zips[,c(4:9)], 4, nstart = 10)
kmean_clust$cluster
table(kmean_clust$cluster, Actual = zips[,2])

# comparing the clusters 
#  Clusters   H-Clust   K-Means
#        1     419         95
#        2     101        141
#        3      18        208
#        4       2         96

remove(list=ls())
