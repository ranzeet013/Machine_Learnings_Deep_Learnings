#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[2]:


dataframe = pd.read_csv('cleaned_crabs.csv')


# In[3]:


dataframe.head()


# In[4]:


dataframe.tail()


# ### Standarizing Data :

# Standardizing is the process of transforming the values of features in a dataset to have a mean of 0 and a standard deviation of 1. This technique is also known as Z-score normalization or standard scaling.

# 
# 
# 1. **Standard Scaling (Z-score normalization):**
#    - Standardizes features by transforming them to have a mean of 0 and a standard deviation of 1.
#    - Formula: \[X_{\text{scaled}} = \frac{X - \text{mean}(X)}{\text{std}(X)}\]
#    - Suitable when the data follows a Gaussian distribution and for algorithms sensitive to the scale of features, such as gradient-based methods.
# 
# 

# In[5]:


data_columns = ['frontal_lobe', 'rear_width', 'carapace_midline', 'maximum_width', 'body_depth']

standarized_dataframe = dataframe.copy()
standarized_dataframe[data_columns] = StandardScaler().fit_transform(standarized_dataframe[data_columns])


# In[6]:


standard_dataframe.head()


# ## Linear Discriminant Analysis ( LDA ) :

# Linear Discriminant Analysis (LDA) is a supervised dimensionality reduction technique used for classification and feature extraction. It aims to find a linear combination of features that characterizes or separates two or more classes in a dataset. The key idea is to maximize the distance between the means of different classes while minimizing the spread (variance) within each class. LDA is often used in the context of pattern recognition and machine learning to improve classification performance by reducing the number of features and emphasizing those that contribute most to class separation.

# ### LDA Analysis on 2 Dimensions :

# In[8]:


lda = LinearDiscriminantAnalysis(n_components = 2)

lda_dataframe = lda.fit_transform(standarized_dataframe[data_columns].values, 
                                  y = standarized_dataframe['class'])
standarized_dataframe[['LDA1', 'LDA2']] = lda_dataframe


# In[10]:


standarized_dataframe.head()


# In[11]:


fig = plt.figure(figsize = (8, 5))
sns.scatterplot(x = standarized_dataframe.LDA1, 
                y = standarized_dataframe.LDA2, 
                hue = 'class', 
                data = standarized_dataframe)


# ### LDA Analysis on 3 Dimensions :

# In[12]:


lda = LinearDiscriminantAnalysis(n_components = 3)

lda_dataframe = lda.fit_transform(standarized_dataframe[data_columns].values, 
                                  y = standarized_dataframe['class'])
standarized_dataframe[['LDA1', 'LDA2', 'LDA3']] = lda_dataframe


# In[14]:


standarized_dataframe.head()


# In[16]:


fig = plt.figure(figsize = (8, 4))

crab_colors = {'bluemale':'red', 
               'bluefemale':'yellow', 
               'orangemale':'blue', 
               'orangefemale':'green'}

ax = fig.add_subplot(111, projection = '3d')

plt.scatter(x = standarized_dataframe.LDA1, 
            y = standarized_dataframe.LDA2, 
            zs = standarized_dataframe.LDA3, 
            c = standarized_dataframe['class'].apply (lambda x: crab_colors[x]), s = 100)


# In[ ]:




