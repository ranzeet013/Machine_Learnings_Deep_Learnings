#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.datasets import fetch_california_housing        #importing california datatset

housing = fetch_california_housing()                         #selecting data and target value
x = housing['data']
y = housing['target']


# In[6]:


from sklearn.model_selection import train_test_split        #splitting datatet in training and testing dataset

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size = 0.2, 
                                                    random_state = 42)

x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[7]:


from sklearn.preprocessing import StandardScaler           #standerizing dataset
scaler = StandardScaler()                              

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# In[8]:


from sklearn.svm import LinearSVR                          #applying linearscr
                                                          
lin_svr = LinearSVR(random_state = 42)
lin_svr.fit(x_train_scaled, y_train)


# In[9]:


from sklearn.metrics import mean_squared_error               #calculating mse

y_pred = lin_svr.predict(x_train_scaled)
mse = mean_squared_error(y_pred, y_train)
mse


# In[11]:


import numpy as np

np.sqrt(mse)


# In[15]:


from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform

param_distributions = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}  #parameter distributions  
random_search_cv = RandomizedSearchCV(SVR(),                                  
                                      param_distributions, 
                                      n_iter = 10, 
                                      verbose = 1, 
                                      cv = 3, 
                                      random_state = 42)
random_search_cv.fit(x_train_scaled, y_train)


# In[16]:


random_search_cv.best_estimator_


# In[18]:


y_pred = random_search_cv.best_estimator_.predict(x_train_scaled)
mse = mean_squared_error(y_train, y_pred)                         #mean square error           
rmse = np.sqrt(mse)                                               #root mean square error
rmse


# In[ ]:




