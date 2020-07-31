rm(list = ls())
library(data.table)
library(caret)
library(dplyr)
library(ggthemes)
library(corrplot)
library(RColorBrewer)
library(rpart)
library(rattle)
library(gbm)
library(MASS)


setwd("C:/Users/aleja/Documents/Cursos/Coursera R pratices/Prediction Assignment Writeup")

train_url<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test_url<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

ifelse(!dir.exists(file.path(getwd(), "Data")), 
       dir.create(file.path(getwd(), "Data")), FALSE)

download.file(url = train_url, destfile = file.path("./Data", "self_movement_train_data.csv"), 
              method = "curl")

download.file(url = test_url, destfile = file.path("./Data", "self_movement_test_data.csv"), 
              method = "curl")

list.files("./Data")

fread("./Data/self_movement_train_data.csv")->train_df
fread("./Data/self_movement_test_data.csv")->test_df

str(train_df)
str(test_df)

sum(is.na(train_df)==TRUE)
sum(is.na(test_df)==TRUE)


train_df[,-c(1:5)] %>%
     select_if(~ !any(is.na(.)))->train_data; sum(is.na(train_data)==TRUE)

test_df[,-c(1:5)] %>%
     select_if(~ !any(is.na(.)))->test_data; sum(is.na(test_data)==TRUE)
str(train_data)
str(test_df)

head(train_data[,-c(1:5)])


class(train_data$classe)
levels(as.factor(train_data$new_window))


g1 <- ggplot(data = train_data, aes(x=as.factor(train_data$classe)))
g1 + geom_bar(fill="firebrick3", colour="black")+theme_stata()+
     ylab("Frequency") + xlab("Classes") + ggtitle("Frequency of different classes")
     
corM <- cor(train_data[, -c(1,55)])
corrplot(corM, order = "FPC", method = "circle", type = "upper", 
         tl.cex = 0.7, tl.col="black", col=brewer.pal(n=8, name="RdBu"))

set.seed(123)

partition  <- createDataPartition(train_data$classe, p=0.75, list=FALSE)

train_set <- train_data[partition, ]

test_set <- train_data[-partition, ]

################
control_rf <- trainControl(method="cv", number=4, verboseIter=FALSE)
fit_rf <- train(classe ~ ., data=train_set, method="rf",
                          trControl=control_rf)

fit_rf$finalModel

predict_rf<- predict(fit_rf, newdata=test_set)
matrix_rf <- confusionMatrix(predict_rf, as.factor(test_set$classe))
matrix_rf

plot(fit_rf)

#############
controlGBM <- trainControl(method = "cv", number = 5)
fit_gbm  <- train(classe ~ ., data=train_set, method = "gbm",
                    trControl = controlGBM, verbose = FALSE)

predict_gbm<- predict(fit_gbm, newdata=test_set)
matrix_gbm <- confusionMatrix(predict_gbm, as.factor(test_set$classe))
matrix_gbm
###########

fit_tree <- rpart(classe ~ ., data=train_set, method="class")

fancyRpartPlot(fit_tree)

fit_tree$control

predict_tree<- predict(fit_tree, newdata=test_set, type="class")
matrix_tree <- confusionMatrix(predict_tree, as.factor(test_set$classe))
matrix_tree$overall[[1]]


#############
control <- trainControl(method = "cv", number = 5)

fit_lda <- train(classe~., data=train_set, 
                 method="lda", metric="Accuracy", trControl=control)

predict_lda <- predict(fit_lda, newdata=test_set)
matrix_lda<- confusionMatrix(predict_lda, as.factor(test_set$classe))

matrix_lda

#######
fit_knn <- train(classe~., data=train_set, method="knn", metric="Accuracy", 
                 trControl=control)

predict_knn <- predict(fit_knn, newdata=test_set)
matrix_knn <- confusionMatrix(predict_knn, as.factor(test_set$classe))
matrix_knn

plot(fit_knn)



perform_acuracy <- c(round(matrix_lda$overall[[1]], 4),
             round(matrix_knn$overall[[1]], 4),
             round(matrix_gbm$overall[[1]], 4),
             round(matrix_rf$overall[[1]], 4),
             round(matrix_tree$overall[[1]], 4))
model_names<-c('Linear Discrimination Analysis (LDA)', 
                         'K- Nearest Neighbors (KNN)',
                         'Gradient Boosting (GBM)',
                         'Random Forest (RF)',
                         'Decision Tree')

(accuracies<-as.data.frame(cbind(model_names, perform_acuracy)))


predict_test_data <- predict(fit_gbm, newdata=test_data)
table(predict_test_data,test_data$problem_id)
