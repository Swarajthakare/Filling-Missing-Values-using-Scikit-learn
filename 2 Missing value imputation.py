#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


# ### Load data set

# In[2]:


train = pd.read_csv(r'B:\Worked dataset\House price prediction advance regration technique\train.csv')
test = pd.read_csv(r'B:\Worked dataset\House price prediction advance regration technique\test.csv')
print('shape of train df =',train.shape)
print('shape of test df =',test.shape)


# In[3]:


train.head()


# In[4]:


X_train = train.drop(columns = 'SalePrice')
y_test = train['SalePrice']
print('shape of X_train df =',X_train.shape)
print('shape of y_test df =',y_test.shape)


# In[5]:


num_vars = X_train.select_dtypes(include = ['int64','float64']).columns
num_vars


# In[6]:


X_train[num_vars].isna().sum()


# In[7]:


imputer_mean = SimpleImputer(strategy = 'mean')
#imputer_class = SimpleImputer(strategy = 'costant',fill_value = 99)


# In[8]:


imputer_mean.fit(X_train[num_vars])


# In[9]:


imputer_mean.statistics_


# In[10]:


imputer_mean.transform(X_train[num_vars])


# In[11]:


X_train[num_vars] =imputer_mean.transform(X_train[num_vars])
test[num_vars] = imputer_mean.transform(test[num_vars])


# In[12]:


X_train[num_vars].isna().sum()


# In[13]:


test[num_vars].isna().sum()


# In[14]:


cat_vars = X_train.select_dtypes(include = 'O').columns
cat_vars


# In[15]:


X_train[cat_vars].isna().sum()


# In[16]:


imputer_mode = SimpleImputer(strategy = 'most_frequent')
#imputer-mode = SimpleImputer(strategy = 'constant',fill_na = 'sea')


# In[17]:


imputer_mode.fit(X_train[cat_vars])


# In[18]:


imputer_mode.statistics_


# In[19]:


imputer_mode.transform(X_train[cat_vars])


# In[20]:


X_train[cat_vars] = imputer_mode.transform(X_train[cat_vars])
test[cat_vars] = imputer_mode.transform(test[cat_vars])


# In[21]:


X_train[cat_vars].isna().sum()


# In[22]:


test[cat_vars].isna().sum()


# In[23]:


X_train.isna().sum().sum()


# In[24]:


test.isna().sum().sum()


# In[ ]:




