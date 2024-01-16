#!/usr/bin/env python
# coding: utf-8

# In[1]:


try:
    from sklearn.datasets import fetch_openml                               #importing dataset
    mnist = fetch_openml('mnist_784', version=1, cache=True)
except ImportError:
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')

x = mnist["data"]
y = mnist["target"]

x_train = x[:60000]
y_train = y[:60000]
x_test = x[60000:]
y_test = y[60000:]


# In[2]:


import numpy as np

np.random.seed(42)
random_idx = np.random.permutation(60000)               #shuffeling datatset

x_train = x_train.iloc[random_idx]
y_train = y_train.iloc[random_idx]


# In[3]:


from sklearn.svm import LinearSVC

lin_clf = LinearSVC(random_state=42)
lin_clf.fit(x_train, y_train)


# In[4]:


from sklearn.metrics import accuracy_score

y_pred = lin_clf.predict(x_train)
accuracy_score(y_train, y_pred)


# In[5]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.astype(np.float32))
x_test_scaled = scaler.transform(x_test.astype(np.float32))


# In[6]:


lin_clf = LinearSVC(random_state=42)
lin_clf.fit(x_train_scaled, y_train)


# In[7]:


y_pred = lin_clf.predict(x_train_scaled)
accuracy_score(y_train, y_pred)


# In[8]:


from sklearn.svm import SVC

svm_clf = SVC(decision_function_shape="ovr", 
              gamma="auto")
svm_clf.fit(x_train_scaled[:10000], y_train[:10000])


# In[9]:


y_pred = svm_clf.predict(x_train_scaled)
accuracy_score(y_train, y_pred)


# In[10]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform

param_distributions = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}
random_search_cv = RandomizedSearchCV(svm_clf, 
                                      param_distributions, 
                                      n_iter=10, 
                                      verbose=2, 
                                      cv=3)
random_search_cv.fit(x_train_scaled[:1000], y_train[:1000])


# In[12]:


random_search_cv.best_estimator_


# In[13]:


random_search_cv.best_estimator_.fit(x_train_scaled, y_train)


# In[14]:


y_pred = random_search_cv.best_estimator_.predict(x_train_scaled)
accuracy_score(y_train, y_pred)


# In[ ]:




