#source('(your path)/Visualization_step2.R')

#3D graphs in R with OpenGL

#Required libraries
rm(list=ls(all=TRUE))

list.of.packages <- c("igraph", "rgl", "xlsx")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

library(igraph)
library(rgl)
library(xlsx)
library(NbClust)
#options(warn=-1)

cat("\n Welcome to Visualization for networks!\n\n") 

# Seed for random numbers generation
set.seed(1)

#Load files
#Ask for network file
cat("\n Please, select the network file\n\n") 
file <- file.choose()
fname<-basename(file)
fdirectory<-dirname(file)

#Set Working directory
setwd(fdirectory)

myNet <- read.table(fname, header = FALSE, sep = "")
coordsAndsc <- read.table("coords-sc.txt", header = FALSE, sep = ",")
coords <- coordsAndsc[,1:3]
sc<-coordsAndsc[,4]
anglesAndGpq <-read.table("angles-Gpq.txt", header = FALSE, sep = ",")
angles <- anglesAndGpq[,1]
Gpq <- anglesAndGpq[,2]
An <- read.table("anglesFull.txt", header = FALSE, sep = ",")
communities_full <- read.table("clusters.txt", header = FALSE, sep = ",")


answer<-invisible(readline(prompt="Do you want to include labels for the nodes? [Y/N]"))
if (answer=="Y" | answer=="y" | answer=="yes" | answer=="Yes"){
  cat("\n Please, select the labels file\n\n") 
  file2 <- file.choose()
  names <- read.table(file2, header = FALSE, sep = ",", stringsAsFactors=FALSE, quote = "")
}else{
  names<-data.frame(seq(from = 1, to = length(sc), by = 1),seq(from = 1, to = length(sc), by = 1))
}

cat("\n Please, wait. \n") 

#Creation of the network
g <- graph_from_adjacency_matrix(as.matrix(myNet), mode=c("undirected"))
degrees<- as.matrix(degree(g))
g <- set.vertex.attribute(g, "sc", value=sc)
g <- set.vertex.attribute(g, "degree", value=degree(g))
g <- set.edge.attribute(g, "communicability_angle", value=angles)
g <- set.edge.attribute(g, "Gpq", value=Gpq)
rbPal <- colorRampPalette(c('paleturquoise1','black'))
Col <- rbPal(10)[as.numeric(cut(E(g)$Gpq,breaks = 10))] #Colors for edges
g <- set.edge.attribute(g, "color", value=Col)

g <- set.vertex.attribute(g, "names", value=names[,2])


# # # # # # # Detecting classical communities with Infomap # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
lec <- cluster_leading_eigen(g)
clusters<-cluster_infomap(g, nb.trials = 100)
g <- set.vertex.attribute(g, "group", value=membership(clusters))
if (length(clusters)!=1){
ColNode <- rainbow(length(clusters))[as.numeric(cut(V(g)$group,breaks = length(clusters)))]
}else{ColNode=rep("#666666",length(sc))}
g <- set.vertex.attribute(g, "color", value=ColNode)



# Plot

#Node sizes
mns <- 5 #Min node size for plotting
Mns <- 20 #Max node size for plotting
a <- (Mns-mns)/(max(sc)-min(sc))
y0 <- Mns - ((Mns-mns)*max(sc))/(max(sc)-min(sc))

#edge sizes
mes <- 1 #Min edge size for plotting
Mes <- 10 #Max edge size for plotting
ae <- (Mes-mes)/(max(Gpq)-min(Gpq))
y0e <- Mes - ((Mes-mes)*max(Gpq))/(max(Gpq)-min(Gpq))

rgl.open()# Open a new RGL device
par3d("windowRect"= c(0,0,1300,1300))
rgl.bg(color = "white") # Setup the background color
#rglplot(g, layout=as.matrix(coords),vertex.color=V(g)$color, vertex.size= y0 + a*V(g)$sc, vertex.label = V(g)$names, vertex.label.color="black", vertex.label.dist=1, edge.width= y0e + ae*E(g)$Gpq, edge.color=E(g)$color)
rglplot(g, layout=as.matrix(coords),vertex.color=V(g)$color, vertex.size= 10, vertex.label = V(g)$names, vertex.label.color="black", vertex.label.dist=1, edge.width= 2, edge.color=E(g)$color)
legend3d("top", legend ="Infomap communities", cex=1, bty = "n")
rgl.viewpoint(  zoom = .8 )

#Axis en 0,0
rgl.lines(c(-1, 1), c(0, 0), c(0, 0), color = "green")
rgl.lines(c(0, 0), c(-1,1), c(0, 0), color = "green")
rgl.lines(c(0, 0), c(0, 0), c(-1,1), color = "green")

#Export to png
rgl.snapshot( 'infomap_clusters.png', fmt = "png", top = TRUE )


my_clusters <- data.frame(name=names[,2],cluster=as.matrix(membership(clusters)))
my_clusters <- my_clusters[order(my_clusters[2]),]
write.xlsx(my_clusters, paste(fdirectory,"/clusters_Infomap.xlsx",sep=""))


# # # # # # #  communicability communities in the full communicability space # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

g <- set.vertex.attribute(g, "group", value=as.numeric(as.matrix(communities_full)))
if (max(communities_full)!=1){
  ColNode <- rainbow(max(communities_full))[as.numeric(cut(as.matrix(communities_full),breaks = max(communities_full)))]
}else{ColNode=rep("#666666",length(sc))}
g <- set.vertex.attribute(g, "color", value=ColNode)

#Node sizes
mns <- 5 #Min node size for plotting
Mns <- 20 #Max node size for plotting
a <- (Mns-mns)/(max(sc)-min(sc))
y0 <- Mns - ((Mns-mns)*max(sc))/(max(sc)-min(sc))

#edge sizes
mes <- 1 #Min edge size for plotting
Mes <- 10 #Max edge size for plotting
ae <- (Mes-mes)/(max(Gpq)-min(Gpq))
y0e <- Mes - ((Mes-mes)*max(Gpq))/(max(Gpq)-min(Gpq))


rgl.open()# Open a new RGL device
par3d("windowRect"= c(0,0,1300,1300))
rgl.bg(color = "white") # Setup the background color
#rglplot(g, layout=as.matrix(coords),vertex.color=V(g)$color, vertex.size= y0 + a*V(g)$sc, vertex.label = V(g)$names,vertex.label.color="black", vertex.label.dist=1, edge.width= y0e + ae*E(g)$Gpq, edge.color=E(g)$color)
rglplot(g, layout=as.matrix(coords),vertex.color=V(g)$color, vertex.size= 10, vertex.label = V(g)$names,vertex.label.color="black", vertex.label.dist=1, edge.width= 2, edge.color=E(g)$color)
#legend3d("top", legend ="Communicability (full) communities", cex=1, bty = "n")
rgl.viewpoint(  zoom = .8 )

#Ejes en 0,0
rgl.lines(c(-1, 1), c(0, 0), c(0, 0), color = "green")
rgl.lines(c(0, 0), c(-1,1), c(0, 0), color = "green")
rgl.lines(c(0, 0), c(0, 0), c(-1,1), color = "green")

#Export to png
rgl.snapshot( 'communicability_full.png', fmt = "png", top = TRUE )


write.xlsx(my_clusters, paste(fdirectory,"/clusters_full.xlsx",sep=""))


cat("\n Succesfully completed. \n") 
options(warn=0)