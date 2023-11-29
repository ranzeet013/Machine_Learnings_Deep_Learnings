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

from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import LocallyLinearEmbedding


# In[2]:


dataframe = pd.read_csv('cleaned_crabs.csv')


# In[3]:


dataframe.head()


# In[4]:


dataframe.tail()


# In[5]:


data_columns = ['frontal_lobe',	'rear_width', 'carapace_midline', 'maximum_width', 'body_depth']

min_max_dataframe = dataframe.copy()
min_max_dataframe[data_columns] = MinMaxScaler().fit_transform(dataframe[data_columns])


# In[6]:


min_max_dataframe.head()


# In[7]:


min_max_dataframe.describe()


# ### Locally Linear Embedding :

# Locally Linear Embedding (LLE) is a nonlinear dimensionality reduction technique. It preserves local relationships between data points, making it effective for capturing nonlinear structures. LLE reconstructs each point as a linear combination of its neighbors and seeks a lower-dimensional representation. It is valuable for manifold learning, revealing the underlying structure of data. LLE is applied in diverse fields, such as image processing, feature extraction, and clustering.

# ### Locally Linear Embedding with 2 Dimensions :

# In[8]:


lle = LocallyLinearEmbedding(n_components = 2, n_neighbors = 15)

lle_dataframe = lle.fit_transform(min_max_dataframe[data_columns])

print('Reconstruction error :', lle.reconstruction_error_)
min_max_dataframe[['LLE1', 'LLE2']] = lle_dataframe


# In[10]:


fig = plt.figure(figsize = (8, 4))
sns.scatterplot(x = 'LLE1',
                y = 'LLE2',
                hue = 'class',
                data = min_max_dataframe)


# ### Locally Linear Embedding with 3 Dimensions :

# In[11]:


lle = LocallyLinearEmbedding(n_components = 3, n_neighbors = 15)

lle_dataframe = lle.fit_transform(min_max_dataframe[data_columns])

min_max_dataframe[['LLE1', 'LLE2', 'LLE3']] = lle_dataframe

print('Reconstruction error:', lle.reconstruction_error_)


# In[12]:


min_max_dataframe.head()


# In[15]:


fig = plt.figure(figsize = (8, 4))
ax = fig.add_subplot(111, projection = '3d')

crab_colors = {'bluemale':'red', 
               'bluefemale':'yellow', 
               'orangemale':'green', 
               'orangefemale':'blue'}

plt.scatter(
    min_max_dataframe.LLE1,
    min_max_dataframe.LLE2,
    zs = min_max_dataframe.LLE3,
    depthshade = False,
    c = dataframe['class'].apply(lambda x: crab_colors[x]),
    s = 100 
)


# In[ ]:




