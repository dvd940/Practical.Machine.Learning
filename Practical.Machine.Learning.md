# Practical Machine Learning Project
October 18, 2015  




## Introduction
In this project, the goal is to use data from accelerometers on the belt, forearm, arm, and dumbbell of 6 participants that were asked to perform barbell lifts correctly and incorrectly in 5 different ways and predict the manner in which they did the exercise. 

The methods used by the participants to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl were

* Exactly according to the specification (Class A)

* Throwing the elbows to the front (Class B) 

* Lifting the dumbbell only halfway (Class C) 

* Lowering the dumbbell only halfway (Class D) 

* Throwing the hips to the front (Class E).

It is the class of method used that will be predicted.

More information on this data set is available from the website here: http://groupware.les.inf.puc-rio.br/har

## Load and Clean The Data
**Note: ** The data is assumed to be located in a sub folder named "data".

If required the data can be downloaded from the following locations

Train Data : https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

Test Data  : https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


```r
train.raw <- read.csv("data/pml-training.csv") #load train data
```

After loading the data features that have no variability are removed since they won't affect the model.


```r
library(caret)
nzv.data <- nearZeroVar(train.raw, saveMetrics = TRUE)  # near zero variance
train <- train.raw[ , nzv.data$nzv == FALSE]
```

Next, features that obviously do not affect the outcome of the models are removed.


```r
drop.cols <- c("X", "user_name", "raw_timestamp_part_1", 
               "raw_timestamp_part_2", "cvtd_timestamp", "num_window")
keep.cols <- names(train[(!names(train) %in% drop.cols)])
train <- train[, keep.cols]
```

Next, features that contain too many NAs will be removed. 


```r
nas <- sapply(train,function(x) sum(is.na(x))) # how many NAs in eaach column
```

From this function, it is observed that some columns contain 19216 NA values. Since the total number of values in each column is 19622, the columns with 19216 NAs will be removed. 


```r
drop.nas <- names(nas[nas == 19216])
keep.cols <- names(train[(!names(train) %in% drop.nas)])
train <- train[, keep.cols]
```

## Build The Models 

### Description of How Final Model Is Built

1. Slice the train data into train and test partitions.

2. Run a number of models using the train partition.

  * Linear Discriminant Analysis (LDA) with Cross Validation.
  
  * Random Forest with Cross Validation.

3. For each model, use the test partition from the data slicing step to determine the accuracy and out of sample error.

4. Select the model with the best accuracy and out of sample error on the test partition.

### Data Slicing

Prior to modelling, the original training data set is split into train and test partitions with a 60/40 split.


```r
train.split <- createDataPartition(y = train$classe, p = 0.6, list = FALSE)
train.data <- train[train.split,]
test.data <- train[-train.split,]
```

### Model 1 : Linear Discriminant Analysis with Cross Validation

Initially **Linear Discriminant Analysis** Model with **Cross Validation** will be created

```r
set.seed(1235)
ldaFit <- train(classe~ .,data=train.data,method="lda", 
                trControl = trainControl(method="cv"),number=3) 
prediction.lda <- predict(ldaFit, test.data)
cf.lda <- confusionMatrix(prediction.lda, test.data$classe)
cf.lda
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1850  238  149   68   51
##          B   50  981  148   56  253
##          C  159  176  881  163  118
##          D  167   54  149  944  141
##          E    6   69   41   55  879
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7055          
##                  95% CI : (0.6952, 0.7155)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.627           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8289   0.6462   0.6440   0.7341   0.6096
## Specificity            0.9099   0.9199   0.9049   0.9221   0.9733
## Pos Pred Value         0.7852   0.6593   0.5885   0.6488   0.8371
## Neg Pred Value         0.9304   0.9155   0.9233   0.9465   0.9172
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2358   0.1250   0.1123   0.1203   0.1120
## Detection Prevalence   0.3003   0.1897   0.1908   0.1854   0.1338
## Balanced Accuracy      0.8694   0.7831   0.7745   0.8281   0.7914
```

```r
lda.accuracy <- round(cf.lda$overall["Accuracy"], 3)  # LDA Accuracy
lda.out.of.sample.error <- as.numeric(1 - lda.accuracy)  # LDA Out Of Sample error
```

From the confusion matrix we can see that the accuracy of this lda model is 0.705 which implies that the **out of sample error** is 0.295. 

### Model 2 : Random Forest with Cross Validation

Next a Random Forest model is tried.


```r
rfFit <- train(classe~ .,data=train.data,method="rf", trControl = trainControl(method="cv"), number=3)
```



To test the accuracy of the prediction, the test subset is used.


```r
prediction.rf <- predict(rfFit, test.data)
cf.rf <- confusionMatrix(prediction.rf, test.data$classe)  # Determine accuracy of model
cf.rf 
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231    6    0    0    0
##          B    1 1510    9    0    0
##          C    0    2 1356   16    1
##          D    0    0    3 1266    0
##          E    0    0    0    4 1441
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9946          
##                  95% CI : (0.9928, 0.9961)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9932          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9996   0.9947   0.9912   0.9844   0.9993
## Specificity            0.9989   0.9984   0.9971   0.9995   0.9994
## Pos Pred Value         0.9973   0.9934   0.9862   0.9976   0.9972
## Neg Pred Value         0.9998   0.9987   0.9981   0.9970   0.9998
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1925   0.1728   0.1614   0.1837
## Detection Prevalence   0.2851   0.1937   0.1752   0.1617   0.1842
## Balanced Accuracy      0.9992   0.9966   0.9941   0.9920   0.9993
```

```r
rf.accuracy <- round(cf.rf$overall["Accuracy"], 3)  # RF Accuracy
rf.out.of.sample.error <- as.numeric(1 - rf.accuracy)  # RF Out Of Sample Error
```

From the confusion matrix we can see that the accuracy of this lda model is 0.995 which implies that the **out of sample error** is 0.005. 

## Model Selection

Based on this result, the Random Forest model will be used for the prediction. 

## Prediction
To make the predictions, the original test set is loaded. 


```r
test.raw <- read.csv("data/pml-testing.csv")
```

This test set must be modified to have a similar format to the training data.


```r
keep.cols <- names(train[, -53]) # drop the classe variable 
keep.cols[53] <- "problem_id"  # add in a problem_id column
test <- test.raw[, keep.cols]  # Match the columns of the training set and drop the others
```
Next, predictions are made and saved in a variable called answers.

```r
answers <- as.character(predict(rfFit, test))  ## generate the prediction on real test data.
answers
```

```
##  [1] "B" "A" "B" "A" "A" "E" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A"
## [18] "B" "B" "B"
```

