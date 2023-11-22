#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


## loading data


# In[3]:


from sklearn.datasets import load_boston


# In[10]:


boston=load_boston()


# In[11]:


type(boston)


# In[12]:


boston.keys()


# In[13]:


## lets check the description of dataset


# In[15]:


print(boston)


# In[16]:


print(boston.DESCR)


# In[17]:


print(boston.data)


# In[18]:


print(boston.target)


# In[20]:


print(boston.feature_names)


# In[21]:


##preparing the data set


# In[25]:


dataset=pd.DataFrame(boston.data,columns=boston.feature_names)


# In[26]:


dataset.head()


# In[27]:


dataset.columns


# In[29]:


dataset.info()


# In[30]:


dataset['price']=boston.target


# In[31]:


dataset.head()


# In[32]:


dataset.info()


# In[33]:


dataset.describe()


# In[34]:


dataset.shape


# In[35]:


##summarizing the stats of the data


# In[36]:


dataset.describe()


# In[37]:


## check the missing value


# In[38]:


dataset.isnull()


# In[39]:


dataset.isnull().sum()


# In[42]:


## exploratory data analysis
## correlarion 
#correlation is very important while working on the linear regression 


# In[43]:


dataset.corr()


# In[50]:


dataset.columns


# In[51]:


dataset.columns


# In[54]:


plt.scatter(dataset['CRIM'],dataset['price'])
plt.xlabel("crime rate")
plt.ylabel("price")


# In[55]:


plt.scatter(dataset['RM'],dataset['price'])
plt.xlabel("RM")
plt.ylabel("price")


# In[56]:


import seaborn as sns
sns.regplot(x="RM",y="price",data=dataset)


# In[57]:



sns.regplot(x="LSTAT",y="price",data=dataset)


# In[58]:


sns.regplot(x="CHAS",y="price",data=dataset)


# In[60]:


sns.regplot(x="PTRATIO",y="price",data=dataset)


# In[61]:


## independent and dependent features


# In[69]:


x=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]


# In[70]:


X


# In[71]:


y


# In[72]:


## train test split


# In[73]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# In[74]:


x_train


# In[75]:


x_test


# In[77]:


## stadardize data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[78]:


x_train=scaler.fit_transform(x_train)


# In[79]:


x_test=scaler.transform(x_test)


# In[80]:


x_train


# In[81]:


x_test


# In[82]:


## model training


# In[84]:


from sklearn.linear_model import LinearRegression


# In[85]:


regression=LinearRegression()


# In[86]:


regression.fit(x_train,y_train)


# In[87]:


## print the coefficients and the intercepts
print(regression.coef_)


# In[88]:


print(regression.intercept_)


# In[89]:


## on which parametere model has been trained
regression.get_params()


# In[90]:


## predicatio with the test data


# In[92]:


reg_pred=regression.predict(x_test)


# In[93]:


reg_pred


# In[94]:


## plot a scatter plot for the prediction
plt.scatter(reg_pred,y_test)


# In[95]:


plt.scatter(y_test,reg_pred)


# In[96]:


## residuals
residuals=y_test-reg_pred


# In[97]:


residuals


# In[98]:


## plot this residuals


# In[99]:


sns.displot(residuals,kind="kde")


# In[102]:


## scatter plot  with respect to predications and residuals
## uniform distribution


# In[103]:


plt.scatter(reg_pred,residuals)


# In[107]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
        
print(mean_squared_error(y_test,reg_pred))
print(mean_absolute_error(y_test,reg_pred))
print(np.sqrt(mean_squared_error(y_test,reg_pred)))
        
    


# In[108]:


## r square and adjusted R square


# In[109]:


from sklearn.metrics import r2_score
score=r2_score(y_test,reg_pred)


# In[110]:


print(score)


# In[111]:


## adjusted r square
1-(1-score)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)


# In[112]:


## new data prediction


# In[ ]:




