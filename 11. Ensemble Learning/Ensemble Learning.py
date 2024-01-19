#!/usr/bin/env python
# coding: utf-8

# ### Importing Dataset :

# In[2]:


import numpy as np

try:
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1)
    mnist.target = mnist.target.astype(np.int64)
except ImportError:
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')


# ### Splitting Dataset :

# splitting dataset into training/validation set (`x_train_val`, `y_train_val`) and a separate test set (`x_test`, `y_test`). A subsequent split further divides the combined set into training (`x_train`, `y_train`) and validation (`x_val`, `y_val`) subsets.

# In[4]:


from sklearn.model_selection import train_test_split

x_train_val, x_test, y_train_val, y_test = train_test_split(mnist.data, 
                                                            mnist.target, 
                                                            test_size = 10000, 
                                                            random_state = 42)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, 
                                                  y_train_val, 
                                                  test_size = 10000, 
                                                  random_state = 42)


# ### Ensemble Learning :

# training an ensemble of machine learning classifiers on a training dataset. The classifiers include a Random Forest, Extra Trees, Support Vector Machine (SVM), and Multi-Layer Perceptron (MLP). Each classifier is configured with specific parameters, and a for loop iterates through the ensemble.

# In[5]:


from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier 

random_forest_clf = RandomForestClassifier(n_estimators = 10, 
                                           random_state = 42)

extra_trees_clf = ExtraTreesClassifier(n_estimators = 10, 
                                       random_state = 42)

svm_clf = LinearSVC(random_state = 42)

mlp_clf = MLPClassifier(random_state = 42)

estimators = [random_forest_clf, extra_trees_clf, svm_clf, mlp_clf]
for estimator in estimators:
    print("Training the", estimator)
    estimator.fit(x_train, y_train)


# In[6]:


estimators_name = [
    ("random_forest_clf", random_forest_clf),
    ("extra_trees_clf", extra_trees_clf),
    ("svm_clf", svm_clf),
    ("mlp_clf", mlp_clf),
]

voting_clf = VotingClassifier(estimators_name)

voting_clf.fit(x_train, y_train)


# In[7]:


voting_clf.score(x_val, y_val)


# In[8]:


voting_clf.estimators_


# In[9]:


voting_clf.score(x_val, y_val)

