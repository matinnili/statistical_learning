library(caret)
df<-read.csv("psych24r.csv")
class(df)
df
df<-df[,c(3,6:29)]
df_2<-df[,c(6:29,32)]
df
colnames(df)<-c("sex" ,"visual_perception","cubes","paper_form_board","flags","general
_information","paragraph_comprehension","sentence_completion", 
                "word_classification","word_meaning","addition","code","counting_
dots","straight_curved_capitals","word_recognition","number_
recognition","figure_recognition","object_number","number_figure",
                "figure_word","deduction","numerical_puzzles","problem_reasoning",
                "series_completion","arithmetic_problems")
colnames(df_2)<-c("visual_perception","cubes","paper_form_board","flags","general
_information","paragraph_comprehension","sentence_completion", 
                "word_classification","word_meaning","addition","code","counting_
dots","straight_curved_capitals","word_recognition","number_
recognition","figure_recognition","object_number","number_figure",
                "figure_word","deduction","numerical_puzzles","problem_reasoning",
                "series_completion","arithmetic_problems","group")

r<-lda(df$sex~.,data=df)
predict_class<-predict(r)$class
table(df$sex,predict_class,dnn=c("actual_group","predicted_group"))
r_2<-lda(df$sex~.,data=df,CV=TRUE)
table(new_df$classdigit,predict_class,dnn=c("actual_group","predicted_group"))
table(df$sex,r_2$class,dnn=c("actual_group","predicted_group"))
df
predict_class
r_3<-qda(df$sex~.,data=df,CV=TRUE)
predict_class<-predict(r_3)$class
table(df$sex,predict_class,dnn=c("actual_group","predicted_group"))



train_samples<-createDataPartition(df$sex,p=.8,list=FALSE)
train_data<-df[train_samples,]
test_data<-df[-train_samples,]
train_lda<-lda(sex~.,data=train_data)
train_predict<-predict(train_lda,test_data)$class
table(test_data$sex,train_predict,dnn=c("actual_group","predicted_group"))
r$scaling
dim(df)
r<-lda(df_2$group~.,data=df_2)
predict_class<-predict(r)$class
table(df_2$group,predict_class,dnn=c("actual_group","predicted_group"))
r$scaling
r_2<-lda(df_2$group~.,data=df_2,CV=TRUE)
table(df_2$group,r_2$class,dnn=c("actual_group","predicted_group"))
table(df_2$group,r_2$class,dnn=c("actual_group","predicted_group"))
df
predict_class
r_3<-qda(df_2$group~.,data=df_2,CV=TRUE)
predict_class<-predict(r_3)$class
table(df_2$group,r_3$class,dnn=c("actual_group","predicted_group"))



train_samples<-createDataPartition(df_2$group,p=.8,list=FALSE)
train_data<-df_2[train_samples,]
test_data<-df_2[-train_samples,]
train_lda<-lda(group~.,data=train_data)
train_predict<-predict(train_lda,test_data)$class
table(test_data$group,train_predict,dnn=c("actual_group","predicted_group"))

