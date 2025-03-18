#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


# In[3]:


train = pd.read_csv(r'B:\Worked dataset\House price prediction advance regration technique\train.csv')
test = pd.read_csv(r'B:\Worked dataset\House price prediction advance regration technique\test.csv')
print('shape of train df =',train.shape)
print('shape of test df =',test.shape)


# In[4]:


train.head()


# In[5]:


X_train = train.drop(columns = 'SalePrice')
y_test = train['SalePrice']
print('shape of X_train df =',X_train.shape)
print('shape of y_test df =',y_test.shape)


# ### Numerical missing value imputation

# In[7]:


num_vars = X_train.select_dtypes(include = ['int64','float64']).columns
num_vars


# In[8]:


X_train[num_vars].isnull().sum()


# In[9]:


imputer_mean = SimpleImputer(strategy = 'mean')
#imputer_mode = SimpleImputer(strategy = 'constant', fill_value = 99)


# In[12]:


imputer_mean.fit(X_train[num_vars])


# In[13]:


imputer_mean.statistics_


# In[21]:


imputer_mean.transform(X_train[num_vars])


# In[24]:


X_train[num_vars] = imputer_mean.transform(X_train[num_vars])
test[num_vars] = imputer_mean.transform(test[num_vars])


# In[25]:


X_train[num_vars].isna().sum()


# In[26]:


test[num_vars].isna().sum()


# ### Categorical missing value imputation (arbitrary value)

# In[28]:


cat_vars = X_train.select_dtypes(include = 'O').columns
cat_vars


# In[29]:


X_train[cat_vars].isna().sum()


# In[31]:


imputer_mode = SimpleImputer(strategy='most_frequent')


# In[32]:


imputer_mode.fit(X_train[cat_vars])


# In[33]:


imputer_mode.statistics_


# In[34]:


imputer_mode.transform(X_train[cat_vars])


# In[39]:


X_train[cat_vars] = imputer_mode.transform(X_train[cat_vars])
test[cat_vars] = imputer_mode.transform(test[cat_vars])


# In[40]:


X_train[cat_vars].isna().sum()


# In[41]:


test[cat_vars].isna().sum()


# In[ ]:




