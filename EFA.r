

getwd()
setwd("C:/Users/matin/Downloads/")
df<-read.csv("psych24r.csv")
class(df)
df
colnames(df)<-c("visual_perception","cubes","paper_form_board","flags","general
_information","paragraph_comprehension","sentence_completion", 
"word_classification","word_meaning","addition","code","counting_
dots","straight_curved_capitals","word_recognition","number_
recognition","figure_recognition","object_number","number_figure",
"figure_word","deduction","numerical_puzzles","problem_reasoning",
"series_completion","arithmetic_problems")
df<-df[,6:29]
df
sapply(1:6, function(f) factanal(df,factors = f,method="mle")$PVAL)
fact<-factanal(df,factors=6,method="mle")
fact$loadings
fact2<-factanal(df,factors=6,method="mle",scores = "regression")$scores
head(fact2,n=10)
df_1<-df[1:156,]
df_1
df_2<-df[156:301,]
df_2
factanal(df_1,factors=6,method="mle")
factanal(df_2,factors=6,method="mle")
