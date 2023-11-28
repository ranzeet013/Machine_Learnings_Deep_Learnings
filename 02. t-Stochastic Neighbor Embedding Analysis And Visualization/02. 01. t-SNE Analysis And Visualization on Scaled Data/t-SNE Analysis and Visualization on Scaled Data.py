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
from sklearn.preprocessing import MinMaxScaler


# In[2]:


dataframe = pd.read_csv('cleaned_crabs.csv')


# In[3]:


dataframe.head()


# In[4]:


dataframe.tail()


# ### Scaling :

# Scaling is a preprocessing step in data analysis and machine learning that involves transforming the numerical values of features into a standardized range. The primary goal is to ensure that all features contribute equally to the analysis, preventing one feature from dominating due to its larger scale.

# 
# 
# 1. **Min-Max Scaling:**
#    - Scales the values of features to a specific range, often between 0 and 1.
#    - Formula: \[X_{\text{scaled}} = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}\]
#    - Useful when the data distribution is not necessarily Gaussian and when features have different ranges.
# 
# 

# In[7]:


data_columns = ['frontal_lobe', 'rear_width', 'carapace_midline', 'maximum_width', 'body_depth']

min_max_dataframe = dataframe.copy()
min_max_dataframe[data_columns] = MinMaxScaler().fit_transform(min_max_dataframe[data_columns])


# In[8]:


min_max_dataframe.head()


# ## t-Stochastic Neighbor Embedding : 

# t-SNE, or t-distributed stochastic neighbor embedding, is a nonlinear dimensionality reduction technique used for visualizing high-dimensional data. It preserves local and global relationships between data points, revealing intricate structures and clusters. The perplexity parameter controls the balance between global and local preservation. t-SNE is widely employed for data exploration and visualization, providing insights into the inherent patterns and similarities within the dataset

# ### t-SNE on Scaled Data :

# ### t-SNE on 2 Dimensions :

# In[9]:


tsne_dataframe = TSNE(n_components = 2, 
                      perplexity = 10, 
                      n_iter = 2000,
                      init = 'random').fit_transform(min_max_dataframe[data_columns])

min_max_dataframe[['TSNE1', 'TSNE2']] = tsne_dataframe


# In[10]:


min_max_dataframe.head()


# In[11]:


figure = plt.figure(figsize = (8, 4))
sns.scatterplot(x = 'TSNE1', 
                y = 'TSNE2', 
                hue = 'class', data = min_max_dataframe)


# ### t-SNE on 3 Dimensions : 

# In[12]:


tsne_dataframe = TSNE(n_components = 3, 
                      perplexity = 10, 
                      n_iter = 2000,
                      init = 'random').fit_transform(min_max_dataframe[data_columns])

min_max_dataframe[['TSNE1', 'TSNE2', 'TSNE3']] = tsne_dataframe


# In[13]:


min_max_dataframe.head()


# In[14]:


fig = plt.figure(figsize = (8, 4))

crab_colors = {'bluemale':'red', 
               'bluefemale':'yellow', 
               'orangemale':'green', 
               'orangefemale': 'blue'}
ax = fig.add_subplot(111, projection = '3d')
plt.scatter(min_max_dataframe.TSNE1, 
            min_max_dataframe.TSNE2, 
            zs = min_max_dataframe.TSNE3, 
            depthshade = False, 
            c = min_max_dataframe['class'].apply(lambda x: crab_colors[x]), s = 100)


# In[ ]:




