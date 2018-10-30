library(caret)
library(doParallel)
library(randomForest)
library(DMwR)
library(CORElearn)
library(pROC)

trainX <- read.csv('train_X.csv', header = F, sep = "\t")
trainY <- read.csv('train_Y.csv', header = T, sep = '\t') ##### binary output -1 and 1
testX <- read.csv('test_X.csv', header = F, sep = '\t')


############################## Churn outcome #############################################

sum(is.na.data.frame(trainX))

#===============Table Manipulation ===============

#removing any columns with more than 90% NA values
trainX <- trainX[,colSums(is.na(trainX))<=0.9*nrow(trainX)]
testX <- testX[,colSums(is.na(testX))<=0.9*nrow(testX)]

#creating training dataframe
training <- trainX
training$churn <- trainY$churn

#=====using subset for local computation (20%) of data; commented out if not applicable======
small_data <- training##[1:sum(0.2*length(training[,1])),]

#removing columns with more that 70% empty values
for(i in names(small_data)){
  if(length(which(small_data[,i] == ""))/length(small_data[,i])>0.7){
    small_data[,i] <- NULL
  }
}

#removing factor columns that have more than 53 levels (model maximum is 53) 
for(i in names(small_data)){
  if(length(levels(small_data[,i])) > 53){
    small_data[,i] <- NULL
  }
}


#removing any observations that have more that 80% missing values
#small_data <- small_data[rowSums(is.na(small_data))<0.80*length(names(small_data)),]

str(small_data)
small_data$V118 <- NULL
small_data$V119 <- NULL

#converting class into factor data type (originally was integer)
small_data$churn <- ifelse(small_data$churn=='1','churn',
                             ifelse(small_data$churn=='-1','stay',NA))
small_data$churn <- as.factor(small_data$churn)
small_data$churn <- factor(as.character(small_data$churn),
                     levels = rev(levels(small_data$churn)))


#running parallel processing
cl <- makeCluster(detectCores()*0.5)
registerDoParallel(cl)

# shuffling dataset
small_data<-small_data[sample(nrow(small_data)),]

#creating training and test set
split <- createDataPartition(small_data$churn, p = 0.75)[[1]]
small_data_train <- small_data[split,]
small_data_test<- small_data[-split,]


#using resampling methods for class inbalance
table(small_data_train$churn)
small_data_train<- SMOTE(churn~., data = small_data_train, k= 5)
table(small_data_train$churn)

# Imputation needed (% of dataframe is NA)
sum(is.na.data.frame(small_data_train))/(dim(small_data_train)[1]*dim(small_data_train)[2])


#impute missing values
small_data_train <- knnImputation(small_data_train)

#feature selection via information gain
info_gain <- attrEval(churn ~., data = small_data_train, estimator="InfGain")
sort(info_gain, decreasing=T)
num_att <- length(info_gain)-20
predictors<-names(sort(info_gain, decreasing=T)[1:num_att])

#===================================Random Forest============================================
#Accuracy, Kappa, the area under the ROC curve,sensitivity and specificity:
fiveStats <- function(...) c(twoClassSummary(...),
                             defaultSummary(...))


## For 10 Cross validations
ctrl <- trainControl(method = "cv",
                     number = 10,
                     classProbs = TRUE,
                     summaryFunction = fiveStats,
                     verboseIter = TRUE)

#Random Forest Model
rfFit <-  caret::train (x=small_data_train[predictors],
                          y=small_data_train[,62],
                          method = "rf",
                          trControl = ctrl,
                          ntrees = 500,
                          preProcess= c('zv', 'nzv', 'center', 'scale'),
                          tuneLength =10,
                          metric='ROC')


print(rfFit)
plot(rfFit)

small_data_test <- knnImputation(small_data_test)

rf_results <- confusionMatrix(predict(rfFit, small_data_test[predictors]), small_data_test[,62])
rf_results

str(testX$V58)
str(small_data_train$V58)
testX$V58 <- NULL


predict(rfFit,testX[predictors[predictors!='V58']])


# compute the performance using probabilities.
prob_results <- predict(rfFit, small_data_test[predictors])

#ROC curve
rf_ROC <- roc(small_data_test[,62], prob_results[,1],levels = levels(small_data_test[,62]))
rf_ROC
plot(rf_ROC, legacy.axes=TRUE, col='black', xlim=c(1,0), ylim=c(0,1),main='Random Forest ROC') 
#save(rfFit, file="RF_model.RData")

topLeft <- coords(rf_ROC, x = "best", ret="threshold",best.method="closest.topleft")
results_topLeft <- factor(ifelse(prob_results[,1] > topLeft,
                                 "churn", "stay"),
                          levels = levels(small_data_test[,62]))

youden <- coords(rf_ROC, x = "best", ret="threshold", best.method="youden")
results_youden <- factor(ifelse(prob_results[,1] > youden,
                                "churn", "stay"),levels = levels(small_data_test[,62]))


tl_accuracy <-  results_topLeft$overall[1]
tl_kappa <- results_topLeft$overall[2] 
tl_sensitivity = results_topLeft$byClass[1]
tl_specificity = results_topLeft$byClass[2]

results_Youden_PostProcess<-NULL
results_topLeft_PostProcess <- NULL

results<-confusionMatrix(results_youden,small_data_test[,62])

results_Youden_PostProcess<-rbind(results_Youden_PostProcess , c(accuracy,kappa,sensitivity,specificity))
print(results_Youden_PostProcess)

results_topLeft_PostProcess<-rbind(results_topLeft_PostProcess , c(accuracy,kappa,sensitivity,specificity))
print(results_topLeft_PostProcess)




#===============================Stochastic Gradient Boosting=================================

gbmGrid <-  expand.grid(interaction.depth = c(10,15,20), 
                        n.trees = (1:30)*50, 
                        shrinkage = 0.1,
                        n.minobsinnode = 20)

set.seed(100)
gbmFit <-  caret::train (x=small_data_train [predictors],
                         y=small_data_train[,62],
                         method = "gbm",
                         trControl = ctrl,
                         preProcess= c('zv', 'center', 'scale'),
                         tuneGrid = gbmGrid,
                         tuneLength =10,
                         verbose = FALSE,
                         metric='ROC')
print(gbmFit)
plot(gbmFit)

gbm_results <- confusionMatrix(predict(gbmFit, small_data_test[predictors]), small_data_test[,62])
gbm_results

# compute the performance using probabilities.
prob_results <- predict(gbmFit, small_data_test[predictors], type = "prob")

#ROC curve
gbm_ROC <- roc(small_data_test[,62], prob_results[,1],levels = levels(small_data_test[,62]))
gbm_ROC
plot(gbm_ROC, legacy.axes=TRUE, col='black', xlim=c(1,0), ylim=c(0,1),main='Generalized Boosted Regression ROC') 

 

#==================================AdaBoost Classification Trees=======================================
#grid <- expand.grid(nIter = floor((1:100) * 50),
                    #method = c("Adaboost.M1", "Real adaboost"))

adaFit <-  caret::train (x=small_data_train [predictors],
                          y=small_data_train[,62],
                          method = "adaboost",
                          trControl = ctrl,
                         #tuneGrid = grid,
                          preProcess= c('zv', 'nzv', 'center', 'scale'),
                          metric='ROC')
print(adaFit)
plot(adaFit)

ada_results <- confusionMatrix(predict(adaFit, small_data_test[predictors]), small_data_test[,62])
ada_results


prob_results <- predict(adaFit, small_data_test[predictors], type = "prob")
predict(adaFit,testX[predictors])

length(levels(testX$V154))
length(levels(small_data_train$V154))
levels(testX$V154) <-levels(small_data_train$V154)
levels(testX$V29) <-levels(small_data_train$V29)

test_y <- predict(adaFit,testX[predictors])
test_y <- ifelse(test_y=='churn','1',
                           ifelse(test_y=='stay','-1',NA))
write.csv(test_y,file = 'churn_testy.csv')


ada_ROC <- roc(small_data_test[,62], prob_results[,1],levels = levels(small_data_test[,62]))
ada_ROC
plot(ada_ROC, legacy.axes=TRUE, col='black', xlim=c(1,0), ylim=c(0,1),main='AdaBoost ROC') 


#===================================K-Nearest Neighbors Model================================

knn_dat_train <- small_data_train
knn_dat_test <- small_data_test

indx <- sapply(knn_dat_train[,1:61], is.factor)
knn_dat_train[indx] <- lapply(knn_dat_train[indx], function(x) as.numeric(x))

indx <- sapply(knn_dat_test[,1:61], is.factor)
knn_dat_test[indx] <- lapply(knn_dat_test[indx], function(x) as.numeric(x))


knnFit <-  caret::train (x=knn_dat_train [predictors],
                         y=knn_dat_train[,62],
                         method = "knn",
                         trControl = ctrl,
                         tuneLength =10,
                         preProcess= c('zv', 'nzv', 'center', 'scale'),
                         metric='ROC')
print(knnFit)
plot(knnFit)


knn_results <- confusionMatrix(predict(knnFit, knn_dat_test[predictors]), knn_dat_test[,62])
knn_results

prob_results <- predict(knnFit, knn_dat_test[predictors], type = "prob")

knn_ROC <- roc(knn_dat_test[,62], prob_results[,1],levels = levels(knn_dat_test[,62]))
knn_ROC
plot(knn_ROC, legacy.axes=TRUE, col='black', xlim=c(1,0), ylim=c(0,1),main='K-Nearest Neighbors ROC') 


#====================================== Averaged Neural Networks============================================
annFit <-  caret::train (x=small_data_train [predictors],
                         y=small_data_train[,62],
                         method = "avNNet",
                         trControl = ctrl,
                         tuneLength =10,
                         preProcess= c('zv', 'nzv', 'center', 'scale'),
                         trace = F,
                         metric='ROC')
print(annFit)
plot(annFit)


ann_results <- confusionMatrix(predict(annFit, small_data_test[predictors]), small_data_test[,62])
ann_results

prob_results <- predict(annFit, small_data_test[predictors], type = "prob")

ann_ROC <- roc(small_data_test[,62], prob_results[,1],levels = levels(small_data_test[,62]))
ann_ROC
plot(ann_ROC, legacy.axes=TRUE, col='black', xlim=c(1,0), ylim=c(0,1),main='Average Neural Network ROC') 



#================================Model Training Comparison===================================
resamps <- resamples(list(RF = rfFit,
                          GBM = gbmFit,
                          ADA = adaFit,
                          KNN = knnFit,
                          ANN = annFit))

#Results
resamps
summary(resamps)
jpeg('model_comp1.jpg')
theme1 <- trellis.par.get()
theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
theme1$plot.symbol$pch = 16
theme1$plot.line$col = rgb(1, 0, 0, .7)
theme1$plot.line$lwd <- 2
trellis.par.set(theme1)
bwplot(resamps, layout = c(5, 1))
dev.off()


jpeg('model_comp2.jpg')
trellis.par.set(caretTheme())
dotplot(resamps, metric = "ROC")
dotplot(resamps, metric = 'Spec')
dev.off()
resamps$metrics
#Difference between models 
jpeg('model_comp3.jpg')
difValues <- diff(resamps)
difValues
summary(difValues)
trellis.par.set(theme1)
bwplot(difValues, layout = c(5, 1))
dev.off()


#=================================Saving Results=======================================
results <- NULL

trapezoid.auc <- function(sensitivity, specificity){
  return((specificity*(1-sensitivity))/2 + 
           (sensitivity*(1-specificity))/2 + 
           (sensitivity*specificity))
}

rf_accuracy <- rf_results$overall[1]
rf_kappa <- rf_results$overall[2]
rf_sensitivity <- rf_results$byClass[1]
rf_specificity <- rf_results$byClass[2]
rf_auc <- trapezoid.auc(rf_sensitivity,rf_specificity)


gbm_accuracy <- gbm_results$overall[1]
gbm_kappa <- gbm_results$overall[2]
gbm_sensitivity <- gbm_results$byClass[1]
gbm_specificity <- gbm_results$byClass[2]
gbm_auc <- trapezoid.auc(gbm_sensitivity,gbm_specificity)

ada_accuracy <- ada_results$overall[1]
ada_kappa <- ada_results$overall[2]
ada_sensitivity <- ada_results$byClass[1]
ada_specificity <- ada_results$byClass[2]
ada_auc <- trapezoid.auc(ada_sensitivity,ada_specificity)

knn_accuracy <- knn_results$overall[1]
knn_kappa <- knn_results$overall[2]
knn_sensitivity <- knn_results$byClass[1]
knn_specificity <- knn_results$byClass[2]
knn_auc <- trapezoid.auc(knn_sensitivity,knn_specificity)

ann_accuracy <- ann_results$overall[1]
ann_kappa <- ann_results$overall[2]
ann_sensitivity <- ann_results$byClass[1]
ann_specificity <- ann_results$byClass[2]
ann_auc <- ada_ROC$auc[1]
ann_auc <- trapezoid.auc(ann_sensitivity,ann_specificity)

results$accuracy <-rbind(rf_accuracy,gbm_accuracy,ada_accuracy,knn_accuracy,ann_accuracy)
results$kappa <- rbind(rf_kappa,gbm_kappa,ada_kappa,knn_kappa,ann_kappa)
results$sensitivity <-rbind(rf_sensitivity,gbm_sensitivity,ada_sensitivity,knn_sensitivity,ann_sensitivity)
results$specificity <-rbind(rf_specificity,gbm_specificity,ada_specificity,knn_specificity,ann_specificity)
results$auc <-rbind(rf_auc,gbm_auc,ada_auc,knn_auc,ann_auc)

results <-data.frame(results)
row.names(results) <- c('RF',"GBM","ADA","KNN","ANN")
results
write.csv(results,file = "results.csv")


#Producing predicted output
test_y <- predict(adaFit,testX[predictors])
test_y <- ifelse(test_y=='churn','1',
                 ifelse(test_y=='stay','-1',NA))
write.csv(test_y,file = 'churn_testy.csv')
