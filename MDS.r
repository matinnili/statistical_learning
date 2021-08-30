install.packages("fields")
library(fields)
df<-read.csv("psych24r.csv")
class(df)
df<-df[,6:29]
colnames(df)<-c("visual_perception","cubes","paper_form_board","flags","general
_information","paragraph_comprehension","sentence_completion", 
                "word_classification","word_meaning","addition","code","counting_
dots","straight_curved_capitals","word_recognition","number_
recognition","figure_recognition","object_number","number_figure",
                "figure_word","deduction","numerical_puzzles","problem_reasoning",
                "series_completion","arithmetic_problems")
df
getwd()
setwd("C:/Users/matin/Downloads")
D<-rdist(df)
dim(D)
t<-D[1:10,1:10]
t[lower.tri(t)]
j<-dist(df)
round(j,digits = 2)
print(j[1:20,2])
as.matrix(j,diag=0)[1:10,1:10]
p<-cmdscale(j,k=24,eig = TRUE)$eig
p_2<-cmdscale(j,k=300,eig=TRUE)
p_2$points
p_2$eig
round(p,digits=3)
result<-cmdscale(j,k=24,eig = TRUE)
result$points
round(result$points[1:10,1:10],digits=2)
max(abs(dist(df))- dist(cmdscale(j,k=24)))
max(abs(prcomp(df)$x)-abs(cmdscale(j,k=24)))
j_2<-dist(df,method="manhattan")
j_2
x_eig<-cmdscale(k=,j_2,eig=TRUE)$eig
round(cumsum(abs(x_eig))/sum(abs(x_eig)),digits=2)
round(cumsum(x_eig^2)/sum(x_eig^2),digits=2)
x<-cmdscale(j_2,eig=TRUE)
x$eig
criteria<-cumsum(abs(x_eig))/sum(abs(x_eig))
length(criteria[criteria<.8])
max(abs(dist(df,method="manhattan")- dist(cmdscale(j_2,k=13),method = "manhattan")))
criteria_2<-cumsum(x_eig[x_eig>0])/sum((x_eig))
round(criteria_2,digits = 2)
information<-sapply(1:300,function(f) max(abs(dist(df,method="manhattan")- dist(cmdscale(j_2,k=f),method = "manhattan"))))
information
length(x_eig[x_eig>3e5])
getDistMethods()
round(mahalanobis(df,colMeans(df),cov(df))[1:20],digits=2)
cat(round(colMeans(df),digits=2))
round(matrix(cov(df[,1:10]),nrow=10,ncol=10),digits=2)
round(cov(df[,1:5]),digits=2)
sapply(1:300,function(f) abs(sum(dist(df,method="manhattan")^2- dist(cmdscale(j_2,k=f),method = "manhattan")^2))/sum(dist(df,method="manhattan")^2))[13]
dist(df,method="manhattan")^2
l<-matrix(summary(df),nrow=24)
drop(cat(round(colMeans(df)),digits=4))
l<-as.matrix(t(summary(df)))
matrix(l,nrow=24,ncol=6)    
k<-sapply(1:14,function(f) median(df[,f]))
cat(matrix(k,nrow=14))
