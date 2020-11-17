# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 04:55:15 2020

@author: mbust
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import figure, plot, xlabel, ylabel, legend, show, boxplot
from sklearn import model_selection, tree
import sklearn.linear_model as lm
from Functions import rlr_validate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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

#Splitting data in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.80, stratify=y)

#Set up a cross-validation on the train data for parameter validation
K = 20
CV = model_selection.KFold(K, shuffle = True, random_state = 10)

for (k, (train_index, test_index)) in enumerate(CV.split(X_train,y_train)):
    
    print('\Outer-Crossvalidation fold: {0}/{1}'.format(k+1,K)) 
    
    X_cv_train = X_train[train_index]
    X_cv_test = X_train[test_index]
    
    y_cv_train = y_train[train_index]
    y_cv_test = y_train[test_index]
    
    N,M = X_cv_train.shape
    
    #----------------------------Standardizing data ---------------------
    
    mu_in = np.mean(X_cv_train, 0)
    sigma_in = np.std(X_cv_train, 0)
    
    # Check if not all values of an attribute are 0, to avoid division by 0
    for i in np.where(sigma_in==0)[0]:
        sigma_in[i] = 1
    
    # Standardizing the data in the cv folds
    X_cv_train = (X_cv_train - mu_in) / sigma_in
    X_cv_test = (X_cv_test - mu_in) / sigma_in
    
    #Selecting the range of lambda + initializing some matrices to store error rate
    lambda_interval = np.logspace(1, 3, 10)
    train_error_rate = np.zeros((len(lambda_interval),K))
    test_error_rate = np.zeros((len(lambda_interval),K))
    
    #Training the model with each potential lambda
    for i in range(0, len(lambda_interval)):
        mdl = LogisticRegression(penalty='l2', C=1/lambda_interval[i])
        
        mdl.fit(X_cv_train, y_cv_train) 
    
        y_train_est = mdl.predict(X_cv_train).T
        y_test_est = mdl.predict(X_cv_test).T
        
        train_error_rate[i,k] = np.sum(y_train_est != y_cv_train) / len(y_cv_train)
        test_error_rate[i,k] = np.sum(y_test_est != y_cv_test) / len(y_cv_test)

#Standardizing data for main model
    
mu_in = np.mean(X_train, 0)
sigma_in = np.std(X_train, 0)

# Check if not all values of an attribute are 0, to avoid division by 0
for i in np.where(sigma_in==0)[0]:
    sigma_in[i] = 1

# Standardizing the data
X_train = (X_train - mu_in) / sigma_in
X_test = (X_test - mu_in) / sigma_in

#Extract the optimal lambda
lambda_opt_idx=np.argmin(test_error_rate.mean(1))
lambda_opt=lambda_interval[lambda_opt_idx]

#Use the optimal lambda to train the main model on the main training dataset
mdlO = LogisticRegression(penalty='l2', C=1/lambda_opt, random_state=100)
mdlO.fit(X_train, y_train)

model_y_est = mdlO.predict(X_test).T
model_error_rate = np.sum(model_y_est != y_test) / float(len(y_test))

#Write the regression coefficients in tables and export the tables to html
import plotly.graph_objects as go

coefficients = np.zeros(len(mdlO.coef_[0]))
for i in range(len(coefficients)):
    coefficients[i]=mdlO.coef_[0][i]
    
figI = go.Figure(data=[go.Table(header=dict(values=['Attributes(I)', 'Regression Coefficient(I)']),
                 cells=dict(values=[attributeNames[:12], coefficients[:12]]))
                     ])
figI.update_layout(width=500, height=800)
figI.write_html("Table5A.html")

figII = go.Figure(data=[go.Table(header=dict(values=['Attributes(II)', 'Regression Coefficient(II)']),
                 cells=dict(values=[attributeNames[12:], coefficients[12:]]))
                     ])
figII.update_layout(width=500, height=800)
figII.write_html("Table5B.html")

print(lambda_opt)
print(model_error_rate)
    
