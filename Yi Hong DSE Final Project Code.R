#clear memory
rm(list = ls())

#install.packages(c("ROCR", "kknn","boot", "corrplot", "rpart", "MASS", "rpart.plot","pls"))

library(rpart)
library(ROCR) 
library(kknn) 
library(boot) 
library(MASS) 
library(e1071) 

setwd("C:/Users/User/Downloads/YH Y1S1/dse1101/Project/Bank_telemarketing_data")
bank = read.csv("bank-additional.csv", sep=";")

#####################################################################################
### 1) data cleaning 
#####################################################################################

#first we turn our y dependent variable into a binary var
bank$y1 = ifelse(bank$y== "yes", 1,0)
bank = subset(bank, select = -c(y))
sum(is.na(bank))
#no NA values or missing values found

#how will our sample size be reduced if we remove unknowns from categorical predictors
unk = which(bank == "unknown", arr.ind = TRUE) 
if_minus_unknowns = bank[-unk,] 
# after removing all the rows with "unknown", we are only left with 3085 rows, down from 4119, 
# that will greatly reduce the size of our training and test set

sum(duplicated(bank)[1:22]) #check for duplicated; none found

bank2 = bank #after cleaning the data, we call it bank2

#####################################################################################
### 2) exploratary data analysis
#####################################################################################

attach(bank2)

#####################################################################################
# 2A) Handling the "loan" variable
#####################################################################################

# Replacing "yes" with 1, "no" with 0, and handling unknown values

bank2$loan = ifelse(bank2$loan == "yes", 1, ifelse(bank2$loan == "no", 0, NA))

# Replacing unknown values in "loan" with the calculated mean
mean_missing_loan = mean(bank2$loan, na.rm = TRUE)
bank2$loan = ifelse(is.na(bank2$loan), mean_missing_loan, bank2$loan)

#####################################################################################
# 2B) Handling the "housing" variable
#####################################################################################

bank2$housing=ifelse(bank2$housing == "yes", 1 ,ifelse(bank2$housing == "no", 0, NA))
mean_missing_housing =mean(bank2$housing, na.rm = TRUE) 

bank2$housing  = ifelse(is.na(bank2$housing),
                           mean_missing_housing, bank2$housing)



#####################################################################################
# 2C) Removing the "Default" variable and "duration" variable
#####################################################################################

sum(default=="yes")
#since there is only 1 row where default=yes, we remove the entire column of defaults as it will not be helpful in our analysis
bank2 = bank2[,-5] 


#####################################################################################
# 2D) Handling the "education" variable 
#####################################################################################

# Identifying and addressing the "illiterate" class
ill = which(bank2 == "illiterate", arr.ind = TRUE)
bank2 = bank2[-3927,]  # Since there's only one observation for "illiterate," we remove it as it might not be useful in the analysis.

# Reordering factor levels and rescaling the "education" predictor
bank2$education = factor(bank2$education, levels = c("basic.4y",
                                                     "basic.6y","basic.9y","high.school",
                                                     "professional.course","university.degree"))
bank2$education = as.numeric(as.factor(bank2$education))
bank2$education = (bank2$education - 1/2) / 7  

# Handling unknown values in education using imputation techniques
blanks = which(is.na(bank2[,4]))  # Identifying unknown values and converting them to NA.
mean_education = mean(bank2$education, na.rm = TRUE)  # Calculating the mean of education, excluding NA values.
bank2$education  = ifelse(is.na(bank2$education),
                          mean_education, bank2$education)  # Replacing unknowns with the calculated mean.

# Fitting a logistic regression model for education
fit_edu = glm(y1 ~ education, data = bank2, family = binomial)
summary(fit_edu)
# The logistic regression model's results suggest that the "education" predictor is highly statistically significant in predicting the binary outcome "y1." 
# This is based on the very low p-value (0.000145), indicating a strong relationship between education and the outcome.

#####################################################################################
# 3) Splitting the dataset
#####################################################################################

set.seed(1010)
tr = sample(1:nrow(bank2),2060)  
training = bank2[tr,]   # Training split
testing = bank2[-tr,]

#####################################################################################
# 4) PCA
#####################################################################################

# Creating dataset 'bank3' containing only numeric data for PCA as it does not work with categorical data
numerics = unlist(lapply(bank2, is.numeric))
bank3 = bank2[, numerics]

# Splitting the dataset into training and testing sets (50/50 split)
pcntrain = round(nrow(bank3) * 0.5)
set.seed(1010)
pc.tr = sample(1:nrow(bank3), pcntrain)  
pc.training = bank3[pc.tr,]  
pc.testing = bank3[-pc.tr,]  

# Performing Principal Component Analysis (PCA) on the training set
prall = prcomp(pc.training, scale = TRUE)

# Producing a biplot using the biplot() function on the prcomp object
biplot(prall, main = "Figure 2: biplot")

prall.s = summary(prall)
prall.s$importance

# Saving the proportion of variance explained for each principal component
scree = prall.s$importance[2,]

# Plotting the scree plot
plot(scree, main = "Figure 1: Scree Plot of Principal Components", xlab = "Principal Component", 
     ylab = "Proportion of Variance Explained", ylim = c(0, 1), type = 'b', cex = 0.8)

# Performing Partial Least Squares Regression (PLS) on the training set
library(pls)
pcr.fit = pcr(y1 ~ . - duration, data = pc.training, scale = TRUE, validation = "CV")

summary(pcr.fit)

# Plotting the Cross-Validated Mean Squared Error (MSEP) during model validation
validationplot(pcr.fit, val.type = "MSEP", main = "LOOCV", legendpos = "topright") 

# Making predictions on the testing set using the PLS model
pcr.pred = predict(pcr.fit, newdata = pc.testing, ncomp = 11, type = 'response')


mean((pc.testing$y1 - pcr.pred)^2)  # MSE for PCR: 0.07865927
pcr_confusion = table(pcr.pred > 0.5, pc.testing$y1)

# Creating a prediction object for ROC curve analysis
pcr_pred = prediction(as.numeric(pcr.pred), pc.testing$y1)

# Creating a performance object for ROC curve analysis
pcr_perf = performance(pcr_pred, measure = "tpr", x.measure = "fpr")

# Plotting the ROC curve
plot(pcr_perf)

# Calculating the Area Under the Curve (AUC) for the ROC curve
pcr_auc = performance(pcr_pred, measure = "auc")
pcr_auc@y.values  # AUC of 0.7660102

#####################################################################################
# 5) Logistic Regression Model (with all var and hand selected var)
#####################################################################################


# Training logistic regression model without 'duration', 'euribor3m', 
glm1_fit = glm(y1 ~ .-duration -euribor3m , data = training, family = "binomial")

# Predicting probabilities on the testing set
glm1_prob = predict(glm1_fit, newdata = testing, type = "response")

# Calculating Mean Squared Error (MSE) for the logistic regression model
mean((testing$y1 - glm1_prob)^2)  # MSE of 0.07818813

# Creating a confusion matrix for model evaluation
glm1_confusion = table(glm1_prob > 0.5, testing$y1)
glm1_confusion

# Creating a prediction object for ROC curve analysis
glm1_pred = prediction(glm1_prob, testing$y1)

# Calculating Area Under the Curve (AUC) for the ROC curve
glm1_auc = performance(glm1_pred, measure = "auc")
glm1_auc@y.values  # AUC is 0.7577059


###############################################################################

# Training "optimal" logistic regression model with selected predictors
glm_fit = glm(y1 ~ age + contact + month + campaign + poutcome + emp.var.rate + 
                cons.price.idx + cons.conf.idx - 1, data = training, family = "binomial")

# Predicting probabilities on the testing set
glm_prob = predict(glm_fit, newdata = testing, type = "response")

mean((testing$y1 - glm_prob)^2)  # MSE of 0.07622236

# Creating a confusion matrix for model evaluation
glm_confusion = table(glm_prob > 0.5, testing$y1)
glm_confusion
glm_pred = prediction(glm_prob, testing$y1)
glm_auc = performance(glm_pred, measure = "auc")
glm_auc@y.values  # AUC is 0.7907123

# Finding the optimal cutoff for maximum accuracy
glm_accuracy_perf = performance(glm_pred, measure = "acc")
plot(glm_accuracy_perf, col = "deeppink3", lwd = 2)
glm_ind = which.max(slot(glm_accuracy_perf, "y.values")[[1]])
glm_cutoff = slot(glm_accuracy_perf, "x.values")[[1]][glm_ind]


# achieving improved  AUC compared to the all variables model


#####################################################################################
# 6) Decision Tree
#####################################################################################


# Building a Classification Decision Tree using rpart
dtree = rpart(y1 ~ . - duration, data = training, method = "class", minsplit = 10, cp = 0.000001, maxdepth = 30)

# Plotting the Cross-Validation Error Profile for different complexity parameters
plotcp(dtree)

# Finding the optimal complexity parameter (cp)
bestcp = dtree$cptable[which.min(dtree$cptable[,"xerror"]), "CP"]

# Pruning the tree to get the optimal tree
besttree = prune(dtree, cp = bestcp)

# Generating predicted probabilities for the testing set
dtreeprob = predict(besttree, newdata = testing)

# Plotting the pruned decision tree
plot(besttree, uniform = TRUE, main = "Regression Tree") 
text(besttree, digits = 4, use.n = TRUE, fancy = FALSE, bg = 'lightblue')

# Creating a prediction object for ROC curve analysis
dtreepred = prediction(dtreeprob[, 2], testing$y1)



# Calculating Area Under the Curve (AUC) for the ROC curve and MSE
dtreeauc = performance(dtreepred, measure = "auc")
dtreeauc@y.values  # AUC of 0.7231165 
mean((testing$y1-dtreeprob[,2])^2) #MSE is 0.07737852

# Visualizing the pruned tree using the rpart.plot library
library(rpart.plot)
rpart.plot(besttree, shadow.col = "gray", type = 5, extra = 2)
title("Figure 3: Regression Tree")

#### tree with duration, nr.employed , euribor3m, pdays removed ###########
dtree2 = rpart(y1 ~ . - duration -nr.employed - euribor3m -pdays, data = training, method = "class", minsplit = 10, cp = 0.000001, maxdepth = 30)

plotcp(dtree2)
bestcp2 = dtree2$cptable[which.min(dtree2$cptable[,"xerror"]), "CP"]
besttree2 = prune(dtree2, cp = bestcp2)

dtreeprob2 = predict(besttree2, newdata = testing)
plot(besttree2, uniform = TRUE, main = "Regression Tree") 
text(besttree2, digits = 4, use.n = TRUE, fancy = FALSE, bg = 'lightblue')

dtreepred2 = prediction(dtreeprob2[, 2], testing$y1)
dtreeauc2 = performance(dtreepred2, measure = "auc")
dtreeauc2@y.values  

library(rpart.plot)
rpart.plot(besttree2, shadow.col = "gray", type = 5, extra = 2)
title("Figure 4: Regression Tree")

#####################################################################################
# 7) K- Nearest Neighbours
###################################################################################

knncv=train.kknn(y1 ~ age+  education + campaign + pdays + previous +emp.var.rate + 
                            cons.price.idx + cons.conf.idx+ nr.employed, 
                          data =training, kmax=100, kernel = "rectangular")


kbest=knncv$best.parameters$k
knnpredcv=kknn(y1 ~ age+ education + campaign + previous + emp.var.rate + 
                 cons.price.idx + cons.conf.idx+ nr.employed,
               training,testing,k=kbest,kernel = "rectangular")

table(knnpredcv$fitted.values>0.5,testing$y1) #confusion matrix
knnpred = prediction(knnpredcv$fitted.values, testing$y1)
knnperf = performance(knnpred, measure = "tpr", x.measure = "fpr")
knn_auc = performance(knnpred, measure = "auc")
knn_auc@y.values# 0.759901
mean((testing$y1-dtreeprob[,2])^2) #MSE is 0.07737852


# changing p = 0.1
table(knnpredcv$fitted.values>0.05,      testing$y1) 
table(knnpredcv$fitted.values>0.10,      testing$y1) 


#####################################################################################
# 8) Naïve Bayes Classifier
###################################################################################

nbtrain1 = naiveBayes(y1 ~ .-duration-euribor3m, data = training)

# Make class predictions on the testing set
nbprob1 = predict(nbtrain1, testing, type = "class")
nbprobraw1 = predict(nbtrain1, testing, type = "raw")

# Calculate Mean Squared Error (MSE)
mse_nb1 = mean((testing$y1 - nbprobraw1[,2])^2)
mse_nb1 # MSE of the model = 0.1139317

nb_confusion1 = table(nbprob1, testing$y1)
nb_confusion1

# Create prediction object for ROC curve
nbpred1 = prediction(nbprobraw1[,2], testing$y1)
nbperf1 = performance(nbpred1, measure = "tpr", x.measure = "fpr")

# Calculate Area Under the Curve (AUC) for ROC curve
nbauc1 = performance(nbpred1, measure = "auc")
nbauc1@y.values  # AUC of the model =0.7779268


