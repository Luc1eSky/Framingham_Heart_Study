#!/usr/bin/env python
# coding: utf-8

# # Framingham Heart Study #

# The "Framingham" heart disease dataset includes over 4,238 records,16 columns and 15 attributes. 
# 
# The goal of the dataset is to predict whether the patient has 10-year risk of future (CHD) coronary heart disease

# In[1]:


#import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[6]:


# read in the data
framingham_data = pd.read_csv('framingham.csv')


# In[7]:


# check data
framingham_data.head()


# In[4]:


len(framingham_data)


# ## Explaining Dataset ##

# In[5]:


framingham_data.info()


# In[13]:


# countplot developing CHD in 10 years by gender
sns.countplot(x = 'TenYearCHD', data=framingham_data, hue='male')
plt.show()


# In[14]:


# check for null values in the data
framingham_data.isna().sum()


# In[16]:


# visualize null values
sns.heatmap(framingham_data.isna())
plt.show()


# In[17]:


# find the % of null values in education column
framingham_data['education'].isna().sum() / len(framingham_data) * 100


# In[18]:


# find the % of null values in glucose column
framingham_data['glucose'].isna().sum() / len(framingham_data) * 100


# In[20]:


sns.displot(x = 'age', data=framingham_data)
plt.show()


# # Data Cleaning #
# 
# Fill the missing values with average value since missing data is negligible

# In[25]:


# fill education column
framingham_data['education'].fillna(framingham_data['education'].mean(), inplace = True)
framingham_data['glucose'].fillna(framingham_data['glucose'].mean(), inplace = True)
framingham_data['cigsPerDay'].fillna(framingham_data['cigsPerDay'].mean(), inplace = True)
framingham_data['BPMeds'].fillna(framingham_data['BPMeds'].mean(), inplace = True)
framingham_data['totChol'].fillna(framingham_data['totChol'].mean(), inplace = True)
framingham_data['BMI'].fillna(framingham_data['BMI'].mean(), inplace = True)
framingham_data['heartRate'].fillna(framingham_data['heartRate'].mean(), inplace = True)


# In[26]:


framingham_data.isna().sum()


# # Data Modelling #
# 
# Building Model using Logistic Regression

# In[29]:


# Separate Dependent and Independent Variables
x = framingham_data[['male','age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes','totChol','sysBP','diaBP','BMI','heartRate','glucose']]
y = framingham_data['TenYearCHD']


# In[30]:


from sklearn.model_selection import train_test_split


# In[31]:


# train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[32]:


# import Logistic Regression package
from sklearn.linear_model import LogisticRegression


# In[33]:


# Fit Logistic Regression
lr = LogisticRegression()


# In[34]:


lr.fit(x_train, y_train)


# In[35]:


# predict
prediction = lr.predict(x_test)


# # Testing #

# In[37]:


# print confusion matrix
from sklearn.metrics import confusion_matrix
pd.DataFrame(confusion_matrix(y_test, prediction), columns=['Predicted No', 'Predicted Yes'], index = ['Actual No', 'Actual Yes'])


# In[38]:


from sklearn.metrics import classification_report
print(classification_report(y_test, prediction))


# In[ ]:




