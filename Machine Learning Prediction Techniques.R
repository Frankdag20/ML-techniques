# Frank D'Agostino
# Code formulated with help from Harvard College ECON50 Resources
# Special thanks to Gregory Bruich, Ph.D for guidance with scripts

rm(list=ls()) # Remove all objects from the environment
cat('\014') # Clear the console

# Set seed for cross validation and random forests
set.seed(123)

# Install packages (if necessary) and load required libraries
if (!require(haven)) install.packages("haven"); library(haven)
if (!require(randomForest)) install.packages("randomForest"); library(randomForest)
if (!require(rpart)) install.packages("rpart"); library(rpart)
if (!require(tidyverse)) install.packages("tidyverse"); library(tidyverse)
if (!require(caret)) install.packages("caret"); library(caret)

#-------------------------------------------------------------------------------
# Load in cleaned dataset and test for random assignment
#-------------------------------------------------------------------------------

# Open set
natal <- read_dta("natality.dta")
head(natal)

# Check for amount of observations
nrow(natal); length(natal)
# There are 50,000 observations with 41 different variables

# Let's validate for random assignment of the training and testing data
genTest <- subset(natal$baby_female, natal$training==0)
genTrain <- subset(natal$baby_female, natal$training==1)
t.test(genTest, genTrain, equal.values = FALSE)
# Insignificant difference means that random assignment is supported

# Let's do it one more time
ageTest <- subset(natal$mom_age, natal$training==0)
ageTrain <- subset(natal$mom_age, natal$training==1)
t.test(ageTest, ageTrain, equal.values=FALSE)
# Once again, insignificant difference with P-value > 0.05

# Now that we are confident random assignment occurred, we can test 
# linear regression as a predictor

#-------------------------------------------------------------------------------
# Regression model
#-------------------------------------------------------------------------------

# Let's run the linear regression looking at birthweight, mom_age, and hypertension variables
head(ageTest); head(ageTrain)

train <- subset(natal, natal$training==1)
test <- subset(natal, natal$training==0)

weightTrain <- subset(natal$birthweight, natal$training==1)
weightTest <- subset(natal$birthweight, natal$training==0)

hyperReg <- lm(birthweight ~ mom_age + mom_gest_hypertension + mom_chronic_hypertension, data=train) 
summary(hyperReg)

# Generate predictions for training data
yhat_reg_train <- predict(hyperReg, newdata=train)
# Average predicted value
mean(yhat_reg_train)

# With our prediction, we can now find the root mean squared error
squared_error_reg_train <- (weightTrain - yhat_reg_train)^2
# As such, below is the root of the means squared error
print(sqrt(mean(squared_error_reg_train, na.rm=TRUE)))
# Our error for the training sample was 599.5163

# Now generate predictions for testing data
yhat_reg_test <- predict(hyperReg, newdata=test)
mean(yhat_reg_test)

# Find RMSE
squared_error_reg_test <- (weightTest - yhat_reg_test)^2
print(sqrt(mean(squared_error_reg_test, na.rm=TRUE)))

# Clearly, the root mean squared error is higher for the testing data than
# the training data, suggesting overfitting

#-------------------------------------------------------------------------------
# Decision Tree
#-------------------------------------------------------------------------------

# Let's try a variety of trees up to depth 25
test_loop<-rep(0, 25)
train_loop<-rep(0, 25)

# Loop over depths from 1-25, estimating tree and storing root mean squared error
for (i in 1:25){
  # Estimate tree of depth i
  reg_i<-rpart(birthweight ~ mom_age + mom_gest_hypertension + mom_chronic_hypertension, data=subset(natal, training==1), maxdepth = i, cp=0,  minsplit = 1, minbucket = 1) 
  
  # In-sample root MSE
  hat1 <- predict(reg_i, newdata=train)
  se1 <- (weightTrain - hat1)^2
  train_loop[i]<-sqrt(mean(se1, na.rm=TRUE))
  
  # Out of sample root MSE
  hat2 <- predict(reg_i, newdata=test)
  se2 <- (weightTest - hat2)^2
  test_loop[i]<-sqrt(mean(se2, na.rm=TRUE))
}

# Set up data for graph
depth <- c(1:25)
dataGraph <- data.frame(test_loop, train_loop, depth)
names(dataGraph)<-c("rmse.test","rmse.train","depth")

# Plot of root mean squared prediction error in 
ggplot(data=dataGraph) +
  # Training data in blue
  geom_point(aes(x=depth, y=rmse.train), colour = "blue") + 
  geom_line(aes(x=depth, y=rmse.train), colour = "blue") +
  # Test data in red
  geom_point(aes(x=depth, y=rmse.test), colour = "red") +
  geom_line(aes(x=depth, y=rmse.test), colour = "red")  +
  # Labels
  theme(legend.position = "none") +
  labs(title = "Predicting Birthweight with a Decision Tree",
       y = "Root Mean Squared Error",
       x = "Maximum Tree Depth",
       colour = "")

# Clearly, too large of a tree depth exhibits overfitting for the testing
# data (red), and a largely reduced error for the training data (blue)
# We must choose an optimal decision tree depth

#-------------------------------------------------------------------------------
# Cross Validating Decision Tree
#-------------------------------------------------------------------------------

# Let's conduct cross validation using rpart()
nataltree <- rpart(birthweight ~ mom_age+mom_gest_hypertension+mom_chronic_hypertension
                  , data=train
                  , control = rpart.control(minsplit =10,minbucket=5, cp=0.001))
nataltree # Text Representation of Tree
plot(nataltree) # Plot tree
text(nataltree) # Add labels to tree
printcp(nataltree) # Print complexity parameter table using cross validation (xerror)

#Generate predictions for training data
yhat_tree <- predict(nataltree, newdata=train)

#-------------------------------------------------------------------------------
# Random Forest
#-------------------------------------------------------------------------------

#Random Forest from 1000 Bootstrapped Samples (ntree=100)
natalforest <- randomForest(birthweight ~ mom_age+mom_gest_hypertension+mom_chronic_hypertension,
                            ntree=1000, 
                            importance=TRUE, 
                            data=train)

# Text Representation of Tree
getTree(natalforest, 250, labelVar = TRUE) 

#Generate predictions for training data
yhat_forest <- predict(natalforest, newdata=train, type="response")
 
#-------------------------------------------------------------------------------
# Compare Root Mean Squared Errors for Training Data
#-------------------------------------------------------------------------------

# Root mean squared error inthe training sample.
p <- 3
RMSE <- matrix(0, p, 1)
RMSE[1] <- sqrt(var(weightTrain - yhat_reg_train, na.rm=TRUE))
RMSE[2] <- sqrt(var(weightTrain - yhat_tree, na.rm=TRUE))
RMSE[3] <- sqrt(var(weightTrain - yhat_forest, na.rm=TRUE))

data_for_graph <- data.frame(RMSE, c("Regression", "DecisionTree", "RandomForest"))  

# Change name of 1st column of df to "RMSE" (Root Mean Squared Error)
names(data_for_graph)[1] <- "RMSE"

# Change name of 2nd column of df to "Method"
names(data_for_graph)[2] <- "Method"

# Basic barplot
p <- ggplot(data=data_for_graph, aes(x=Method, y=RMSE)) +
  geom_bar(stat="identity")
# Horizontal bar plot
p + coord_flip()

# All are extremely similar RMSE for training data
print(RMSE)

#-------------------------------------------------------------------------------
# Compare Root Mean Squared Errors for Testing Data
#-------------------------------------------------------------------------------

# Get predictions for test data
yhat_forest <- predict(natalforest, newdata=test, type="response")
yhat_tree <- predict(nataltree, newdata=test)
head(yhat_reg_test)

#Calculate RMSE for test data
p <- 3
RMSE <- matrix(0, p, 1)
RMSE[1] <- sqrt(var(weightTest - yhat_reg_test, na.rm=TRUE))
RMSE[2] <- sqrt(var(weightTest - yhat_tree, na.rm=TRUE))
RMSE[3] <- sqrt(var(weightTest - yhat_forest, na.rm=TRUE))

data_for_graph <- data.frame(RMSE, c("Regression", "DecisionTree", "RandomForest"))  

# Change name of 1st column of df to "RMSE" (Root Mean Squared Error)
names(data_for_graph)[1] <- "RMSE"

# Change name of 2nd column of df to "Method"
names(data_for_graph)[2] <- "Method"

# Basic barplot
p <- ggplot(data=data_for_graph, aes(x=Method, y=RMSE)) +
  geom_bar(stat="identity")
# Horizontal bar plot
p + coord_flip()

# Once again, all are extremely similar for testing data
print(RMSE)

