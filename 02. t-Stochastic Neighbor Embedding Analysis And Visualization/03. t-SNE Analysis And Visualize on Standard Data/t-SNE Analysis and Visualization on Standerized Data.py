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

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# In[2]:


dataframe = pd.read_csv('cleaned_crabs.csv')


# In[3]:


dataframe.head()


# In[4]:


dataframe.tail()


# ### Standerizing :

# 
# 
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

standard_dataframe = dataframe.copy()
standard_dataframe[data_columns] = StandardScaler().fit_transform(standard_dataframe[data_columns])


# In[6]:


standard_dataframe.head()


# ## t-Stochastic Neighbor Embedding :

# t-SNE, or t-distributed stochastic neighbor embedding, is a nonlinear dimensionality reduction technique used for visualizing high-dimensional data. It preserves local and global relationships between data points, revealing intricate structures and clusters. The perplexity parameter controls the balance between global and local preservation. t-SNE is widely employed for data exploration and visualization, providing insights into the inherent patterns and similarities within the dataset

# ### t-SNE on Standerized Data :

# ### t-SNE on 2 Dimensions :

# In[7]:


tsne_dataframe = TSNE(n_components = 2, 
                      perplexity = 10, 
                      n_iter = 2000,
                      init = 'random').fit_transform(standard_dataframe[data_columns])

standard_dataframe[['TSNE1', 'TSNE2']] = tsne_dataframe


# In[8]:


standard_dataframe.head()


# In[9]:


figure = plt.figure(figsize = (8, 4))
sns.scatterplot(x = 'TSNE1', 
                y = 'TSNE2', 
                hue = 'class', data = standard_dataframe)


# ### t-SNE on 3 Dimensions :

# In[10]:


tsne_dataframe = TSNE(n_components = 3, 
                      perplexity = 10, 
                      n_iter = 2000,
                      init = 'random').fit_transform(standard_dataframe[data_columns])

standard_dataframe[['TSNE1', 'TSNE2', 'TSNE3']] = tsne_dataframe


# In[11]:


standard_dataframe.head()


# In[12]:


fig = plt.figure(figsize = (8, 4))

crab_colors = {'bluemale':'red', 
               'bluefemale':'yellow', 
               'orangemale':'green', 
               'orangefemale': 'blue'}
ax = fig.add_subplot(111, projection = '3d')
plt.scatter(standard_dataframe.TSNE1, 
            standard_dataframe.TSNE2, 
            zs = standard_dataframe.TSNE3, 
            depthshade = False, 
            c = standard_dataframe['class'].apply(lambda x: crab_colors[x]), s = 100)


# In[ ]:




