#load libraries
library(caret)

########################### Functions ###########################

# Write model predictions to files to submit to Coursera project
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("output/problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

#################################################################


train.raw <- read.csv("data/pml-training.csv")  #load training data

# Remove features with non variance 
nzv.data <- nearZeroVar(train.raw, saveMetrics = TRUE)
train <- train.raw[ , nzv.data$nzv == FALSE]

# Remove features that won't contribute to the model
drop.cols <- c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "num_window")
keep.cols <- names(train[(!names(train) %in% drop.cols)])
train <- train[, keep.cols]

# Remove features that contain too many NA values
nas <- sapply(train,function(x) sum(is.na(x)))  # we find that there are 41 features that have 19216 NAs. 
drop.nas <- names(nas[nas == 19216])
keep.cols <- names(train[(!names(train) %in% drop.nas)])
train <- train[, keep.cols]


# Split Data into Train and Test partitions
train.split <- createDataPartition(y = train$classe, p = 0.6, list = FALSE)  # a 60/40 split
train.data <- train[train.split,]
test.data <- train[-train.split,]


# Linear Discriminant Analysis Model with cross validation
set.seed(1235)
ldaFit <- train(classe~ .,data=train.data,method="lda", 
                trControl = trainControl(method="cv"),number=3) 
prediction.lda <- predict(ldaFit, test.data)
cf.lda <- confusionMatrix(prediction.lda, test.data$classe)
lda.accuracy <- round(cf.lda$overall["Accuracy"], 3)  # LDA Accuracy
lda.out.of.sample.error <- as.numeric(1 - lda.accuracy)  # LDA Out Of Sample error

# Random Forest Model with Cross Validation
# This model takes a while to complete on this computer so the model is saved to an rds file
# and read back in when needed.
# rfFit <- train(classe~ .,data=train.data,method="rf", trControl = trainControl(method="cv"),number=3)
# saveRDS(rfFit, "rf.model.rds")
rfFit <- readRDS("rf.model.rds")


# rfFit.2 <- readRDS("rf.model.2.rds")
prediction.rf <- predict(rfFit, test.data)  # Make prediction using random forest
cf.rf <- confusionMatrix(prediction.rf, test.data$classe)  # Determine accuracy of model
rf.accuracy <- round(cf.rf$overall["Accuracy"], 3)  # RF Accuracy
rf.out.of.sample.error <- as.numeric(1 - rf.accuracy)  # RF Out Of Sample Error





# Use test Data to generate predictions using Random Forest model
test.raw <- read.csv("data/pml-testing.csv")  # read in the test data

# Coerce the test set to match the format of the training set.
keep.cols <- names(train[, -53]) # drop the classe variable 
keep.cols[53] <- "problem_id"  # add in a problem_id column
test <- test.raw[, keep.cols]  # Match the columns of the training set and drop the others

answers <- as.character(predict(rfFit, test))  ## generate the prediction on real test data.
answers
#pml_write_files(answers)  # write the prediction results to files to submit to Coursera


# Out of Sample Error

# ::TODO::

