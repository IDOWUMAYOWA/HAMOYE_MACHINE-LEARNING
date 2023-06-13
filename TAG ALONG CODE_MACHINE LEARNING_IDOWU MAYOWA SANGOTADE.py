#!/usr/bin/env python
# coding: utf-8

# In[242]:


#Import all relevant Packages
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from category_encoders import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.utils.validation import check_is_fitted
 


# In[243]:


# Load the datasets using pandas
df = pd.read_csv("energydata_complete.csv")
df.head()


# ### Quiz 12:  R^2 value to 2dp for a Linear Model between T2 & T6

# In[244]:


#Declare independent Variable x = T2
feature_df = df[["T2"]]
# comfirm we have selected the right column
feature_df.head()


# In[245]:


# Select target or dependent variable T6
target_df = df['T6']


# In[246]:


#Instantiate our linear model
linear_model = LinearRegression()


# In[247]:


#Fit the model for the parameters selected
linear_model.fit(feature_df, target_df)


# In[248]:


pred_2 = linear_model.predict(feature_df)


# In[249]:


#Estimate the r2 score as required
r2_score = r2_score(target_df, pred_2)
round(r2_score, 2)


# ### Quiz 13: Normalise Data, run a Multiple linear regression and estimate the Mean absolute error

# In[250]:


# Drop the columns date and lights
df_new = df.drop(columns = ['date', 'lights'])
#print the new dataframe after dropping the columns
df_new.head()


# In[251]:


#Instantiate the standardization operator
scaler = MinMaxScaler()


# In[252]:


#apply scaler on the dataframe, fit & transform simultaneously
norm_df = pd.DataFrame(scaler.fit_transform(df_new), columns = df_new.columns)
norm_df.head()


# In[253]:


#Select the input and output variable for the models
feat_df = norm_df.drop(columns = 'Appliances')
targ_df = norm_df['Appliances']


# In[254]:


# let's see our predictor dataframe
feat_df.head()


# In[255]:


#Split the dataset using random test split with 70;30 ratio
x_train,x_test,y_train,y_test = train_test_split(feat_df, targ_df, test_size = 0.3, random_state = 42)


# In[256]:


x_train.columns


# In[257]:


#Instantiate the linearmodel
norm_model =LinearRegression()


# In[258]:


norm_model.fit(x_train, y_train)


# In[259]:


#make predictions on the x test values
predicted_values = norm_model.predict(x_test)


# In[260]:


#Estimate the mean absolute error
mae = mean_absolute_error(y_test, predicted_values)


# In[261]:


round(mae, 2)


# ### Quiz 14: Normalise Data, run a Multiple linear regression and estimate the Residuals sum of squares¶

# In[262]:


#Estimate the RSS 
residual = np.sum(np.square(predicted_values - y_test))
round(residual, 2)


# ### Quiz 15: Normalise Data, run a Multiple linear regression and estimate the Root mean squared error¶

# In[263]:


#Estimate root mean squared error to 3dp
root_mean = np.sqrt(mean_squared_error(y_test, predicted_values))
round(root_mean,3)


# ### Quiz 16: Normalise Data, run a Multiple linear regression and estimate the Coefficient of Determination

# In[314]:


# Estimate coefficeient of determination to 2dp 
r2_score = r2_score(y_test, predicted_values)
round(COD, 2)


# ### Quiz 17: Normalise Data, run a Multiple linear regression and Get the Feature weights

# In[268]:


intercept = norm_model.intercept_
#obtain the Feature weights
coefficients = norm_model.coef_


# In[187]:


coefficients


# In[294]:


Regression_Data = pd.Series(coefficients, index = feat_df.columns).sort_values()
#Check Values for both max & min weights on the Regression equation
Regression_Data


# ### Train with Ridge Regression

# In[297]:


Ridge_model =Ridge(alpha = 0.4)
Ridge_model.fit(x_train, y_train)
predicted_values_ridge = Ridge_model.predict(x_test)
root_mean_ridge = np.sqrt(mean_squared_error(y_test, predicted_values_ridge))


# In[298]:


#Check for equality with the linear model
root_mean_ridge == root_mean


# ### Train with Lasso Regression

# In[303]:


Lasso_model =Lasso(alpha = 0.001)
Lasso_model.fit(x_train, y_train)
predicted_values_Lasso = Lasso_model.predict(x_test)


# In[304]:


Coefficient_Las = Lasso_model.coef_


# In[311]:


#Estimate the number of non_zeroes
Non_Zeroes =  len(Coefficient_Las) -((Coefficient_Las == 0).sum())
Non_Zeroes


# ### 20 Estimate the RMSE

# In[312]:


#Estimate the RMSE for the lasso Regression
root_mean_lasso = np.sqrt(mean_squared_error(y_test, predicted_values_Lasso))
round(root_mean_lasso,3)


# In[ ]:





# In[ ]:




