## TRAINING AND TEST RMSE AFTER APPLYING LOG TRANSFORMATIONS FOR 3 METHODS
library(ggplot2)
library(vcd) # Visualizing Categorical Data – Mosaic Plot
library(caTools) #train-test split
library(GoodmanKruskal)#Goodman and Kruskal’s Plot
library(corrplot) #pearson’s correlation plot
library(caret)
library(scales)
library(vip)
library(nnet)
library(e1071)
library(caTools)
library(MLmetrics)
library(car)
library(dplyr)
#install.packages("ROSE")
library(ROSE)

#install.packages("smotefamily")
library(smotefamily)
library(class)
library(e1071)
library(glmnet)

house_price=read.csv("C:/Users/HIRUNI/Desktop/3_2/ST 3082 - Statistical Learning I/ADVANCED ANALYSIS/archive(11)/data.csv")
str(house_price)

house_price=subset(house_price,price!=0) #remove rows having 0 prices.


colSums(house_price==0)
##### No more unimportant Zero values presence
house_price=subset(house_price,select = -c(date,country))
house_price=subset(house_price,select = -c(street))
#house_price=subset(house_price,select = -c(statezip))
#house_price=subset(house_price,select = -c(city))
str(house_price)
dim(house_price)

library(dplyr)
# Convert the 'statezip' column in df to numeric by extracting the numeric part
house_price <- house_price %>%
  mutate(statezip = as.numeric(sub("\\D+", "", statezip)))
# Print the updated data frame
# 
#  house_price$renovated = ifelse(house_price$yr_renovated > 0,"yes","no")
# # # Remove the old variable (yr_renovated)
#  house_price=subset(house_price,select = -yr_renovated) #remove old var (yr_renovated)
#  #house_price$renovated
#  str(house_price)

# house_price$built_yr = ifelse(house_price$yr_built > 1976,"new","old")
# house_price=subset(house_price,select = -yr_built)
# str(house_price)



#house_price$bedrooms=factor(house_price$bedrooms)
#house_price$bathrooms=factor(house_price$bathrooms)
#house_price$floors=factor(house_price$floors)
#house_price$waterfront=factor(house_price$waterfront)
#house_price$view=factor(house_price$view)
#house_price$condition=factor(house_price$condition)
#house_price$yr_built=factor(house_price$yr_built)
 house_price$city=factor(house_price$city)
 house_price$statezip=factor(house_price$statezip)

library(readr)
dataset=house_price #cleaned full dataset


#splitting to training and testing
#install.packages("caTools")
library(caTools)
set.seed(1234) 
sample <- sample.split(house_price$price, SplitRatio = 0.8)
train <- subset(house_price, sample == TRUE)
test <- subset(house_price, sample == FALSE)
write.csv(train, file = "train.csv", row.names = FALSE)
write.csv(test, file = "test.csv", row.names = FALSE)
dim(train)
dim(test)
attach(train)
summary(price) 

########################################################################################################################
### Run without outliers
# inter quatile method to remove outliers 
Q1=quantile(train$price,0.25)
Q3=quantile(train$price,0.75)
IQR=IQR(train$price)
train=subset(train,train$price>(Q1-1.5*IQR) & train$price<(Q3+1.5*IQR))
#This line subsets the train dataset, keeping only the rows where the price variable falls within the range of Q1 - 1.5 * IQR to Q3 + 1.5 * IQR. This is a common technique to remove outliers from the dataset.


library(glmnet)# fitting generalized linear models with regularization.
library(caret)#  provides a unified interface for training and testing various machine learning models
library(recipes)#  preprocessing and feature engineering in machine learning workflows.
#install.packages("vip")
library(vip)#Variable Importance in Predictive models. I

x1=model.matrix(price~.,train)[,-1] #Train set
#creates the design matrix x1 for the model. It converts the train dataset into a matrix format suitable for modeling. The formula price~. indicates that the price variable is the response variable, and all other variables in the dataset are used as predictors. The [, -1] part removes the intercept term from the design matrix.

#summary(x1)


##Uncomment to run log transform 
y1=log(train$price)

x2=model.matrix(price~.,test)[,-1] #Test set

#summary(x1)
#summary(x2)
#Uncomment to run log transform 
y2=log(test$price)

dim(x1); dim(x2)



#---------------Ridge--------------------------------
fit.ridge=glmnet(x1,y1,alpha = 0)
plot(fit.ridge,xvar = "lambda",label = TRUE,lw=2)
#legend("topright", legend = colnames(x1), col = 1:ncol(x1), lwd = 2)

#Doing cross validation to select the best lambda
set.seed(1)
cv.ridge=cv.glmnet(x1,y1,alpha=0)

plot(cv.ridge)
bestlam=cv.ridge$lambda.min
bestlam

#Fitting the ridge regression model under the best lambda
model1=glmnet(x1,y1,alpha = 0,lambda = bestlam)
best.fit.ridge=glmnet(x1,y1,alpha = 0)
coefficients=coef(best.fit.ridge,s = bestlam)
ridge.pred=predict(best.fit.ridge,newx = x2,s = bestlam)
#Uncomment to run log transform
test_RMSE=RMSE(ridge.pred,y2)
test_RMSE
ridge.pred2=predict(best.fit.ridge,newx = x1,s = bestlam)
train_rmse <- RMSE(ridge.pred2, y1)
train_rmse
#RMSE(exp(ridge.pred),exp(y2))
df=data.frame(as.matrix(coef(best.fit.ridge,s = bestlam)))
df1=data.frame(rownames(df),df[,1])
df1[1,1]="Intercept"
saveRDS(df1,"ridge.coef.rds") #Saved for Web App



#---------------Lasso--------------------
fit.lasso=glmnet(x1,y1,alpha = 1)
plot(fit.lasso,xvar = "lambda",label = TRUE,lw=2)
#legend("topright", legend = colnames(x1), col = 1:ncol(x1), lwd = 2)
#Doing cross validation to select the best lambda
set.seed(1)
cv.lasso=cv.glmnet(x1,y1,alpha=1)
plot(cv.lasso)
bestlam=cv.lasso$lambda.min
bestlam
#Fitting the lasso regression model under the best lambda
model2=glmnet(x1,y1,alpha = 0,lambda = bestlam)
best.fit.lasso=glmnet(x1,y1,alpha = 1)
coef(best.fit.lasso,s = bestlam)
lasso.pred=predict(best.fit.lasso,newx = x2,s = bestlam)
#RMSE(lasso.pred,y2)
test_RMSE=RMSE(lasso.pred,y2)
test_RMSE
lasso.pred=predict(best.fit.lasso,newx = x1,s = bestlam)
#RMSE(lasso.pred,y2)
test_RMSE=RMSE(lasso.pred,y1)
test_RMSE


#Uncomment to run log transform 
#RMSE(exp(lasso.pred),exp(y2))
### Elastic net
set.seed(1)
cv.elasticnet = train(x=x1,y=y1,method="glmnet",preProc=c("zv","center","scale"),trControl=trainControl(method="cv", number = 10),tuneLength=10)
cv.elasticnet$bestTune
ggplot(cv.elasticnet)
best.fit.elastic=glmnet(x1,y1,alpha = cv.elasticnet$bestTune$alpha)
coef(best.fit.elastic,s = cv.elasticnet$bestTune$lambda)
elastic.pred=predict(best.fit.elastic,newx = x2,s =cv.elasticnet$bestTune$lambda)
test_Rmse=RMSE(elastic.pred,y2)
test_Rmse


elastic.pred=predict(best.fit.elastic,newx = x1,s =cv.elasticnet$bestTune$lambda)
train_Rmse=RMSE(elastic.pred,y1)
train_Rmse



























