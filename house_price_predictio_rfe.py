# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 23:07:12 2021

@author: jysethy
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split

housing = pd.read_csv("C:/Users/jysethy/krishnaik/ML-Python/Housing.csv")


#Data Preparation
#List of variables to be mapped

varlist =  ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

# Defining the map function
def binary_map(x):
    return x.map({'yes': 1, "no": 0})

# Applying the function to the housing list
housing[varlist] = housing[varlist].apply(binary_map)


# Applying Dummy for furninshing status as it has got there diff. categories

status = pd.get_dummies(housing['furnishingstatus'],drop_first= True)

# Add results to housing dataset and remove furnishing staus form original dataframe

housing = pd.concat([housing,status], axis=1)

housing.drop(['furnishingstatus'],axis=1, inplace=True)


#Splitting the data into train and test

df_train,df_test = train_test_split(housing, train_size=0.70,random_state=100)


#Rescaling of features using min-max scaler

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
        
# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables
num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])

## Dividing X, Y split

y_train = df_train.pop('price')
X_train = df_train

##BUilding of model using RFE feature selection

# Importing RFE and LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# Running RFE with the output number of the variable equal to 10
lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, 10)             # running RFE
rfe = rfe.fit(X_train, y_train)

list(zip(X_train.columns,rfe.support_,rfe.ranking_))

col = X_train.columns[rfe.support_]


##Building the model using status mpodel

X_train_rfe = X_train[col]

# Adding a constant variable 
import statsmodels.api as sm  
X_train_rfe = sm.add_constant(X_train_rfe)

lm = sm.OLS(y_train,X_train_rfe).fit()

print(lm.summary())


##Bedroom is insignificant hence removing from dataframe

X_train_new = X_train_rfe.drop(['bedrooms'],axis=1)

#Adding a constant variabel

X_train_lm = sm.add_constant(X_train_lm)

lm = sm.OLS(y_train,X_train_lm).fit()

print (lm.summary())

# Calculate the VIFs for the new model
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


##Residual Analysis

y_train_price = lm.predict(X_train_lm)

# Importing the required libraries for plots.
# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)                         # X-label

#Making prediction on test set
#Apply scaler for test set

num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']

df_test[num_vars] = scaler.transform(df_test[num_vars])

y_test = df_test.pop('price')
X_test = df_test

# Now let's use our model to make predictions.

# Creating X_test_new dataframe by dropping variables from X_test
X_train_new = X_train_new.drop(['const'],axis=1)

X_test_new = X_test[X_train_new.columns]
# Adding a constant variable 
X_test_new = sm.add_constant(X_test_new)

# Making predictions
y_pred = lm.predict(X_test_new)

#Model Evaluation

# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)    