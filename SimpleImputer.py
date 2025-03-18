#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


# In[2]:


train = pd.read_csv(r'B:\Worked dataset\House price prediction advance regration technique\train.csv')
test = pd.read_csv(r'B:\Worked dataset\House price prediction advance regration technique\test.csv')


# In[3]:


print('Shape of train df =',train.shape)
print('Shape of trian df =',test.shape)


# In[4]:


train.head()


# In[5]:


X_train = train.drop(columns = 'SalePrice')
y_test = train['SalePrice']
print('Shape of X_train df =',X_train.shape)
print('Shape of y_test df =',y_test.shape)


# ### Numeric var imputation

# In[6]:


num_vars = X_train.select_dtypes(include = ['int64','float64']).columns
num_vars


# In[7]:


X_train[num_vars].isnull().sum()


# In[8]:


imputer_mean = SimpleImputer(strategy = 'mean')
#imputer_mean = SimpleImputer(strategy = 'constant',fill_value = 99)


# In[9]:


imputer_mean.fit(X_train[num_vars])


# In[10]:


imputer_mean.statistics_


# In[11]:


imputer_mean.transform(X_train[num_vars])


# In[12]:


X_train[num_vars] = imputer_mean.transform(X_train[num_vars])
test[num_vars] = imputer_mean.transform(test[num_vars])


# In[13]:


X_train[num_vars].isna().sum()


# In[14]:


test[num_vars].isna().sum()


# ### Categorical variable imputation

# In[15]:


cat_vars = X_train.select_dtypes(include = 'O').columns
cat_vars


# In[16]:


X_train[cat_vars].isna().sum()


# In[17]:


imputer_mode = SimpleImputer(strategy = 'most_frequent')
#imputer_mode = SimpleImputer(strategy = 'most_frequent',fill_value = 'sea')


# In[18]:


imputer_mode.fit(X_train[cat_vars])


# In[19]:


imputer_mode.statistics_


# In[21]:


imputer_mode.transform(X_train[cat_vars])


# In[22]:


X_train[cat_vars] = imputer_mode.transform(X_train[cat_vars])
test[cat_vars] = imputer_mode.transform(test[cat_vars])


# In[24]:


X_train.isna().sum().sum()


# In[26]:


X_train.head()


# In[ ]:




