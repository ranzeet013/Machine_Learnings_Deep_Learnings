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

from sklearn.manifold import Isomap
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
# 1. **Min-Max Scaling:**
#    - Scales the values of features to a specific range, often between 0 and 1.
#    - Formula: \[X_{\text{scaled}} = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}\]
#    - Useful when the data distribution is not necessarily Gaussian and when features have different ranges.
# 
# 

# In[5]:


data_columns = ['frontal_lobe', 'rear_width', 'carapace_midline', 'maximum_width', 'body_depth']

min_max_dataframe = dataframe.copy()
min_max_dataframe[data_columns] = MinMaxScaler().fit_transform(min_max_dataframe[data_columns])


# In[6]:


min_max_dataframe.head()


# ## ISOMAP :

# ISOMAP (Isometric Mapping) is a dimensionality reduction technique aiming to preserve the intrinsic geometry of high-dimensional data. It constructs a neighborhood graph based on Euclidean distances, approximates geodesic distances using graph shortest paths, and embeds the data in a lower-dimensional space. Key parameters include the number of neighbors and target dimensionality. ISOMAP is useful for visualizing high-dimensional data and capturing nonlinear relationships.

# ### ISOMAP on 2 Dimensions :

# In[8]:


isomap = Isomap(n_components = 2, 
                n_neighbors = 10)

isomap_dataframe = isomap.fit_transform(min_max_dataframe[data_columns])

min_max_dataframe[['ISOMAP1', 'ISOMAP2']] = isomap_dataframe


# In[9]:


min_max_dataframe.head()


# In[11]:


print('Reconstruction error :', isomap.reconstruction_error())


# In[10]:


fig = plt.figure(figsize = (8, 5))
sns.scatterplot(x = 'ISOMAP1', 
                y = 'ISOMAP2', 
                hue = 'class', 
                data = min_max_dataframe)


# ### ISOMAP on 3 Dimensions :

# In[12]:


isomap = Isomap(n_components = 3, 
                n_neighbors = 10)

isomap_dataframe = isomap.fit_transform(min_max_dataframe[data_columns])

min_max_dataframe[['ISOMAP1', 'ISOMAP2', 'ISOMAP3']] = isomap_dataframe


# In[13]:


min_max_dataframe.head()


# In[16]:


crab_colors = {'bluemale':'blue', 
               'bluefemale':'red', 
               'orangemale':'yellow', 
               'orangefemale':'green'}
fig = plt.figure(figsize = (8, 5))

ax = fig.add_subplot(111, projection = '3d')

plt.scatter(x = min_max_dataframe.ISOMAP1, 
            y = min_max_dataframe.ISOMAP2, 
            zs = min_max_dataframe.ISOMAP3, 
            depthshade = False, 
            c = min_max_dataframe['class'].apply(lambda x: crab_colors[x]), s = 100)


# In[ ]:




