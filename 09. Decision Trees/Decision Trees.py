#!/usr/bin/env python
# coding: utf-8

# In[10]:


from sklearn.datasets import make_moons     

x, y = make_moons(n_samples=1000,                    #make sample dataset
                  noise=0.4, 
                  random_state=42)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,                   #splitting the dataset in train_test 
                                                    test_size = 0.2, 
                                                    random_state = 42)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

params = {'max_leaf_nodes': list(range(2, 100)),                             #hyperparameter tuning with grid search
          'min_samples_split': [2, 3, 4]}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42),          #applying decision tree classifier
                           params, 
                           n_jobs = -1, 
                           verbose = 1, cv = 3)

grid_search.fit(x_train, y_train)                                            #training 


# In[11]:


grid_search.best_estimator_           #best estimators 


# In[13]:


from sklearn.metrics import accuracy_score                   #accuracy score

y_pred = grid_search.predict(x_test)
accuracy_score(y_test, y_pred)

