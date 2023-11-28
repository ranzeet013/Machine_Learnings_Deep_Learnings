#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

from sklearn.manifold import TSNE


# In[3]:


dataframe = pd.read_csv('cleaned_crabs.csv')


# In[4]:


dataframe.head()


# In[5]:


dataframe.tail()


# ## t-Stochastic Neighbor Embedding :

# t-SNE, or t-distributed stochastic neighbor embedding, is a nonlinear dimensionality reduction technique used for visualizing high-dimensional data. It preserves local and global relationships between data points, revealing intricate structures and clusters. The perplexity parameter controls the balance between global and local preservation. t-SNE is widely employed for data exploration and visualization, providing insights into the inherent patterns and similarities within the dataset

# ### t-SNE on Raw Data :

# ### t-SNE On 2 Dimensions :

# In[10]:


data_columns = {'frontal_lobe',  'rear_width', 	'carapace_midline',	 'maximum_width', 'body_depth'}

tsne_dataframe = TSNE(n_components = 2, 
                      perplexity = 10, 
                      n_iter = 2000,
                      init = 'random').fit_transform(dataframe[data_columns])

dataframe[['TSNE1', 'TSNE2']] = tsne_dataframe


# In[11]:


dataframe.head()


# In[12]:


figure = plt.figure(figsize = (8, 4))
sns.scatterplot(x = 'TSNE1', 
                y = 'TSNE2', 
                hue = 'class', data = dataframe)


# ### t-SNE on 3 Dimensions :

# In[14]:


tsne_dataframe = TSNE(n_components = 3, 
                      perplexity = 10, 
                      n_iter = 2000,
                      init = 'random').fit_transform(dataframe[data_columns])

dataframe[['TSNE1', 'TSNE2', 'TSNE3']] = tsne_dataframe


# In[15]:


dataframe.head()


# In[19]:


fig = plt.figure(figsize = (8, 4))

crab_colors = {'bluemale':'red', 
               'bluefemale':'yellow', 
               'orangemale':'green', 
               'orangefemale': 'blue'}
ax = fig.add_subplot(111, projection = '3d')
plt.scatter(dataframe.TSNE1, 
            dataframe.TSNE2, 
            zs = dataframe.TSNE3, 
            depthshade = False, 
            c = dataframe['class'].apply(lambda x: crab_colors[x]), s = 100)


# In[ ]:




