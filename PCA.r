df<-read.csv("psych24r.csv")
class(df)
df
df<-df[,c(6:29)]
colnames(df)<-c("visual_perception","cubes","paper_form_board","flags","general
_information","paragraph_comprehension","sentence_completion", 
                "word_classification","word_meaning","addition","code","counting_
dots","straight_curved_capitals","word_recognition","number_
recognition","figure_recognition","object_number","number_figure",
                "figure_word","deduction","numerical_puzzles","problem_reasoning",
                "series_completion","arithmetic_problems")
df
library(factoextra)
pca<-princomp(df,cor=TRUE,scores = TRUE)
pca_2<-prcomp(df,scale=TRUE)
cumsum(pca$sdev)/sum(pca$sdev)
fviz_eig(pca_2,ncp = 24)
cumsum(pca_2$sdev)/sum(pca_2$sdev)
round(cumsum(eigen(cor(df))$values)/sum(eigen(cor(df))$values),3)
round(pca$loadings,2)
round(eigen(cor(df))$values,3)
pairs(pca$scores[,1:10])
round(cor(pca$scores[,1:10]),4)
fviz_pca_var(pca,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)
