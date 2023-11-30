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

from sklearn.manifold import MDS
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

# In[5]:


data_columns = ['frontal_lobe', 'rear_width', 'carapace_midline', 'maximum_width', 'body_depth']

min_max_dataframe = dataframe.copy()
min_max_dataframe[data_columns] = MinMaxScaler().fit_transform(min_max_dataframe[data_columns])


# In[6]:


min_max_dataframe.head()


# ##  Multidimensional Scaling (MDS) : 

# 
# MDS (Multidimensional Scaling) is applied to a dataset using scikit-learn. It scales the data, calculates pairwise dissimilarities, and reduces dimensionality to 2 components. The resulting MDS components are added to the DataFrame, and the stress value, indicating how well the reduced representation preserves original distances, is printed.

# ### MDS on 2 Dimensions :

# In[8]:


mds = MDS(n_components = 2,
          n_init = 15,
          metric = True)
mds_dataframe = mds.fit_transform(min_max_dataframe[data_columns])

stress_value = mds.stress_

min_max_dataframe[['MDS1', 'MDS2']] = mds_dataframe

print('MSE : ', stress_value)


# In[9]:


min_max_dataframe.head()


# In[10]:


figure = plt.figure(figsize = (8, 4))
sns.scatterplot(x = 'MDS1', 
                y = 'MDS2', 
                hue = 'class', data = min_max_dataframe)


# ### MDS on 3 Dimensions :

# In[11]:


mds = MDS(n_components = 3,
          n_init = 15,
          metric = True)
mds_dataframe = mds.fit_transform(min_max_dataframe[data_columns])

stress_value = mds.stress_

min_max_dataframe[['MDS1', 'MDS2', 'MDS3']] = mds_dataframe

print('MSE : ', stress_value)


# In[12]:


min_max_dataframe.head()


# In[13]:


fig = plt.figure(figsize = (8, 4))

crab_colors = {'bluemale':'red', 
               'bluefemale':'yellow', 
               'orangemale':'green', 
               'orangefemale': 'blue'}
ax = fig.add_subplot(111, projection = '3d')
plt.scatter(min_max_dataframe.MDS1, 
            min_max_dataframe.MDS2, 
            zs = min_max_dataframe.MDS3, 
            depthshade = False, 
            c = min_max_dataframe['class'].apply(lambda x: crab_colors[x]), s = 100)


# In[ ]:




