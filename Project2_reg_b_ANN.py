# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 14:20:55 2020

@author: cleml
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats

#CHANGE optimizer = torch.optim.Adam(net.parameters(),weight_decay=1e-5)
#IN train_neural_net BY REMOVING weight_decay TO NOT DO REGULARIZATION

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
X = df.drop(['thalach'], axis = 1).to_numpy()
y = df[['thalach']].to_numpy().reshape((len(X),1))
#y = data[:,len(data[0,:])-1].reshape((len(X),1))


N, M = X.shape
C = 2

# Parameters for neural network classifier
#n_hidden_units = np.arange(9,21,2)     # number of hidden units
n_hidden_units = np.arange(1,24,1)
n_replicates = 1        # number of networks trained in each k-fold
max_iter = 100000       # stop criterion 2 (max epochs in training)

# K-fold crossvalidation
K = 10                   # only five folds to speed up this example
CV = model_selection.KFold(K, shuffle=True, random_state=12)
# Make figure for holding summaries (errors and learning curves)
#summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']


mse = np.zeros((K,len(n_hidden_units)))
opt_n_hidden_units = np.zeros(K)
errors = [] # make a list for storing generalizaition error in each loop
for k, (train_index, test_index) in enumerate(CV.split(X,y)): 
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))  
    
    X_train = X[train_index,:]
    X_test = X[test_index,:]
    
    y_train = y[train_index]
    y_test = y[test_index]
    
    # Normalize data
    #X = stats.zscore(X);
    mu_ = np.mean(X_train, 0)
    sigma_ = np.std(X, 0)
        
    # Check if not all values of an attribute are 0, to avoid division by 0
    for i in np.where(mu_==0)[0]:
        sigma_[i] = 1
            
        # Standardizing the data
        X_train = (X_train - mu_) / sigma_
        X_test = (X_test - mu_) / sigma_
    
    # Extract training and test set for current CV fold, convert to tensors
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)
    
    # Training neural networks for various numbers of hidden units
    for i in range(0,len(n_hidden_units)):
            
        # Defining structure of neural network
            
        model = lambda: torch.nn.Sequential(
            # Test for comparison with mult linear regression                
            #torch.nn.Linear(M, 1), #M features to n_hidden_units
            torch.nn.Linear(M, n_hidden_units[i]), #M features to n_hidden_units
            #torch.nn.Tanh(),   # 1st transfer function,
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_units[i], 1), # n_hidden_units to 1 output neuron
            # no final tranfer function, i.e. "linear output"
            )
        loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
        
        # Train the net on training data
        net, final_loss, learning_curve = train_neural_net(model,
                                                   loss_fn,
                                                   X=X_train,
                                                   y=y_train,
                                                   n_replicates=n_replicates,
                                                   max_iter=max_iter)
        print('\n\tBest loss: {}\n'.format(final_loss))

        # Determine estimated class labels for test set
        y_test_est = net(X_test)
        
        # Determine errors and errors
        se = (y_test_est.float()-y_test.float())**2 # squared error
        mse[k,i] = (sum(se).type(torch.float)/len(y_test)).data.numpy() #
    # Train the net on training data
    
    opt_n_hidden_units[k] = n_hidden_units[np.argmin(np.mean(mse,axis=0))]
    
best_n_hidden_units = n_hidden_units[np.argmin(np.mean(mse,axis=0))]
    

mean_error_for_h = np.mean(mse,axis=0)
stand_dev_for_h = np.std(mse,axis=0)
pcs = np.arange(0,len(n_hidden_units),1)
legendStrs = [str(e) + ' hidden unit' for e in n_hidden_units]
#c = ['r','g','b']
bw = 1

plt.figure()
for i in pcs:    
    plt.bar(1+i*bw, mean_error_for_h[i],yerr=stand_dev_for_h[i], width=bw)
    #plt.bar(1+i*bw, mean_error_for_h[i], width=bw, color=palette[i])
plt.xticks(n_hidden_units)
plt.xlabel('Number of hidden units')
plt.ylabel('Average mean squared test error')
#plt.legend(legendStrs, ncol=2, loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()
plt.savefig('C:/Users/cleml/Documents/02450 Introduction to Machine Learning and Data Mining/Project 2/Bar_chart.pdf',bbox_inches='tight')
#plt.title('NanoNose: PCA Component Coefficients')
plt.show()

'''
plt.figure()
plt.title('Optimal lambda: 1e{0}'.format(best_n_hidden_units)),fontsize=12)
plt.plot(n,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
plt.xlabel('Regularization factor',fontsize=12)
plt.ylabel('Squared error (crossvalidation)',fontsize=12)
plt.legend(['Train error','Validation error'],fontsize=12)
plt.grid()
plt.savefig('H_hidden_units.pdf',bbox_inches='tight')
plt.show()
'''

'''
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    
    print('\n\tBest loss: {}\n'.format(final_loss))
    
    # Determine estimated class labels for test set
    y_test_est = net(X_test)
    #y_sigmoid = net(X_test)
    #y_test_est = (y_sigmoid>.5).type(dtype=torch.uint8)

    # Determine errors and errors
    #y_test = y_test.type(dtype=torch.uint8)

    #e = y_test_est != y_test
    e = (y_test_est.float()-y_test.float())**2 # squared error
    error_rate = (sum(e).type(torch.float)/len(y_test)).data.numpy()
    errors.append(error_rate) # store error rate for current CV fold 
    
    # Display the learning curve for the best net in the current fold
    h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
    h.set_label('CV fold {0}'.format(k+1))
    summaries_axes[0].set_xlabel('Iterations')
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss')
    summaries_axes[0].set_title('Learning curves')
    
# Display the error rate across folds
summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(errors)), color=color_list)
summaries_axes[1].set_xlabel('Fold');
summaries_axes[1].set_xticks(np.arange(1, K+1))
summaries_axes[1].set_ylabel('Error rate');
summaries_axes[1].set_title('Test misclassification rates')

print('Diagram of best neural net in last fold:')
weights = [net[i].weight.data.numpy().T for i in [0,2]]
biases = [net[i].bias.data.numpy() for i in [0,2]]
tf =  [str(net[i]) for i in [0,2]]
draw_neural_net(weights, biases, tf, attribute_names=attributeNames)

# Print the average classification error rate
print('\nGeneralization error/average error rate: {0}%'.format(round(100*np.mean(errors),4)))
'''