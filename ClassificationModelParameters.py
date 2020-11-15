# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 20:18:19 2020

@author: mbust
"""

#In this script we are doing the test runs to select 
#a complexity-controling parameter
#
#If there is some error about some NaN value, or a division by 0,
#just re-run the script until that error is gone
#it is due to a particular case where the train/test data has no values
#other than 0

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import figure, plot, xlabel, ylabel, legend, show, boxplot
from sklearn import model_selection, tree
import sklearn.linear_model as lm
from Functions import rlr_validate
from decimal import Decimal

"""
--------------------------------------------
-------PHASE 1 - GETTING DATA READY---------
--------------------------------------------
"""
# Loading data
filename = 'heart.csv'
df = pd.read_csv(filename)

# Making one-of-K encoding of categorical data
a = pd.get_dummies(df['cp'], prefix = "cp")
b = pd.get_dummies(df['thal'], prefix = "thal")
c = pd.get_dummies(df['slope'], prefix = "slope")
d = pd.get_dummies(df['restecg'], prefix = "restecg")

# Extracting target subdataframe from data frame
targ = df.pop('target')

# Concatenating current data frame and encoded one-of-k values and
# making sure target is the  last attribute of the data frame
frames = [df, a, b, c, d, targ]
df = pd.concat(frames, axis = 1)

# Dropping original form of one-of-K encoded variables from dataframe
df = df.drop(columns = ['cp', 'thal', 'slope', 'restecg'])

# Getting attributes names
attributeNames = list(df.columns)
attNb = len(attributeNames)

# splitting dataframe into X and y
X = df.drop(['target'], axis = 1).to_numpy()
y= targ.to_numpy()
N,M = X.shape

"""
------------------------------------------------------------
----PHASE 2 - CLASSIFICATION TREE (COMPLEXITY PARAMETER)----
------------------------------------------------------------
"""
#Check for complexity-controlling parameter with a CV fold
# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True, random_state=12)

# Tree complexity parameter - constraint on maximum depth
tc = np.arange(2, attNb, 1)

# Initialize variables
Error_train = np.empty((len(tc),K))
Error_test = np.empty((len(tc),K))


k=0
#Starting the Cross-Validation
for train_index, test_index in CV.split(X):
    print('Computing CV fold: {0}/{1}..'.format(k+1,K))

    # extract training and test set for current CV fold
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    
    # Standardize outer fold based on training set
    mu = np.mean(X_train, 0)
    sigma = np.std(X_train, 0)
    sigmabis=np.copy(sigma)
    
    for i in np.where(sigma==0)[0]:
        sigma[i] = 1
            
    
    X_train = (X_train - mu ) / sigma
    X_test = (X_test - mu ) / sigma
    
    for i, t in enumerate(tc):
        # Fit decision tree classifier, Gini split criterion, different pruning levels
        dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
        dtc = dtc.fit(X_train,y_train.ravel()) #is sometimes problematic (NaN input)
        y_est_test = dtc.predict(X_test)
        y_est_train = dtc.predict(X_train)
        # Evaluate misclassification rate over train/test data (in this CV fold)
        misclass_rate_test = np.sum(y_est_test != y_test) / float(len(y_est_test))
        misclass_rate_train = np.sum(y_est_train != y_train) / float(len(y_est_train))
        Error_test[i,k], Error_train[i,k] = misclass_rate_test, misclass_rate_train
    k+=1

#Plot results
f = figure()
boxplot(Error_test.T)
xlabel('Model complexity (max tree depth)')
ylabel('Test error across CV folds, K={0})'.format(K))


f = figure()
plot(tc, Error_train.mean(1))
plot(tc, Error_test.mean(1))
xlabel('Model complexity (max tree depth)')
ylabel('Error (misclassification rate, CV K={0})'.format(K))
legend(['Error_train','Error_test'])
    
show()

#print the best results
best_error_test_index=np.argmin(Error_test.mean(1))
best_error_test_value=Error_test.mean(1)[best_error_test_index]
best_error_test_parameter=tc[best_error_test_index]

print("CLASSIFICATION TREE: ")
print('Best error achieved is {:} for the parameter: {:}.'.format(best_error_test_value,best_error_test_parameter))

"""
------------------------------------------------------------
----PHASE 3 - LOGISTIC REGRESSION (COMPLEXITY PARAMETER)----
------------------------------------------------------------
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#Clean the dataset for a regression
#Why drop thalach?
df_reg = df.drop(['thalach'], axis = 1)
attributeNames_reg = list(df_reg.columns)

X_reg = df_reg.to_numpy()
y_reg = df[['thalach']].to_numpy().squeeze()

# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True, random_state=12)

k=0
Y_test = []
Y_test_est = []

# Fit regularized logistic regression model to training data to predict 
# the type of wine
#lambda_interval = np.logspace(-8, 2, 50)
lambda_interval = np.logspace(-4, 6, 50)
train_error_rate = np.empty((len(lambda_interval),K))
test_error_rate = np.empty((len(lambda_interval),K))
coefficient_norm = np.empty((len(lambda_interval),K))

for train_index, test_index in CV.split(X):
    print('Computing CV fold: {0}/{1}..'.format(k+1,K))

    # extract training and test set for current CV fold
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    
    # Standardize outer fold based on training set
    mu = np.mean(X_train, 0)
    sigma = np.std(X_train, 0) 
    
    for i in np.where(sigma==0)[0]:
            sigma[i] = 1
   
    X_train = (X_train - mu ) / sigma
    X_test = (X_test - mu ) / sigma
    
    for j in range(0, len(lambda_interval)):
        mdl = LogisticRegression(penalty='l2', C=1/lambda_interval[j], random_state=12)
        
        mdl.fit(X_train, y_train) #is sometimes problematic (input NaN)
    
        y_train_est = mdl.predict(X_train).T
        y_test_est = mdl.predict(X_test).T
        
        Y_test.append(y_test)
        Y_test_est.append(y_test_est)
        train_error_rate[j,k] = np.sum(y_train_est != y_train) / len(y_train)
        test_error_rate[j,k] = np.sum(y_test_est != y_test) / len(y_test)
    
        w_est = mdl.coef_[0] 
        coefficient_norm[j] = np.sqrt(np.sum(w_est**2))
    k+=1
    
opt_lambda_idx=np.argmin(test_error_rate.mean(1))
min_error=test_error_rate.mean(1)[opt_lambda_idx]
opt_lambda=lambda_interval[opt_lambda_idx]


fig=plt.figure(figsize=(8,8))
#plt.plot(np.log10(lambda_interval), train_error_rate*100)
#plt.plot(np.log10(lambda_interval), test_error_rate*100)
#plt.plot(np.log10(opt_lambda), min_error*100, 'o')
plt.semilogx(lambda_interval, test_error_rate.mean(1)*100)
plt.semilogx(lambda_interval, test_error_rate.mean(1)*100)
plt.semilogx(opt_lambda, min_error*100, 'o')
plt.text(1, 3, "Minimum test error: " + str(np.round(min_error*100,2)) + ' % at 1e' + str(np.round(np.log10(opt_lambda),2)),fontsize=12)
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$', fontsize=12)
plt.ylabel('Error rate (%)', fontsize=12)
plt.title('Classification error', fontsize=12)
plt.legend(['Training error','Test error','Test minimum'],loc='upper right',fontsize=12)
plt.ylim([0, 30])
plt.grid()
#plt.tight_layout()
plt.show()  
fig.savefig('plot.pdf')  
"""
plt.figure(figsize=(8,8))
plt.semilogx(lambda_interval, coefficient_norm,'k')
plt.ylabel('L2 Norm')
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.title('Parameter vector L2 norm')
plt.grid()
plt.show()    
"""
print("LOGISTIC REGRESSION: ")
lambda_answer=f"{opt_lambda:.2E}"
print('Best error achieved is {:} for the parameter: {:}.'.format(min_error*100, lambda_answer))
