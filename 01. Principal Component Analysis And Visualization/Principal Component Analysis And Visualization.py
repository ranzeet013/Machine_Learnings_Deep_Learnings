#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

pd.set_option('display.precision', 3)

import warnings
warnings.filterwarnings('ignore')

from sklearn import set_config
set_config(display = 'text')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[2]:


dataframe = pd.read_csv('cleaned_crabs')


# In[3]:


dataframe.head()


# In[4]:


dataframe.tail()


# In[6]:


data_columns = ['frontal_lobe', 'rear_width', 'carapace_midline', 'maximum_width', 'body_depth']
dataframe_standerized = dataframe.copy()

dataframe_standerized[data_columns] = StandardScaler().fit_transform(dataframe[data_columns])
dataframe_standerized.describe().T


# ### PCA :

# 
# PCA is a dimensionality reduction method that transforms high-dimensional data into a lower-dimensional space by identifying orthogonal axes capturing maximum variance. It utilizes eigenvalues and eigenvectors to determine principal components. The resulting components are uncorrelated, allowing for efficient representation of data patterns. PCA is commonly used for visualization, noise reduction, and feature extraction in data analysis and machine learning. Its application aids in improving computational efficiency and mitigating overfitting risks.

# In[8]:


pca = PCA()
pca.fit(dataframe_standerized[data_columns])

print(pca.explained_variance_ratio_)

print(pca.explained_variance_ratio_.cumsum())


# In[10]:


fig = plt.figure(figsize = (8, 4))

plt.plot(range(1, len(pca.singular_values_) + 1), pca.singular_values_, alpha = 0.8, marker = '.')
y_label = plt.ylabel('Eigen Values')
x_labels = plt.xlabel('Components')
plt.title('PCA Screenplot')


# ### Explained Variance by Components :

# In[15]:


fig = plt.figure(figsize=(8, 4))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha = 0.8, marker = '.', label = 'Explained Variance')
y_label = plt.ylabel('Explained Variance')
x_label = plt.xlabel('Components')

plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), c = 'r', marker = '.', label = 'Cumulative Explained Variance')
plt.legend()
plt.title('Percentage of Explained Variance by Component')
plt.show()


# In[19]:


sns.heatmap(pca.components_, cmap = 'seismic', xticklabels = list(dataframe.columns[3:-1]),
            vmin = np.max(np.abs(pca.components_)),
            vmax = np.max(np.abs(pca.components_)),
            annot = True)


# ### PCA: Transformation And Visualization 

# In[20]:


transformed_dataframe = pca.transform(dataframe_standerized[data_columns])

dataframe_standerized[['PC1', 'PC2', 'PC3']] = transformed_dataframe[:,:3]



# In[21]:


dataframe_standerized.head()


# In[24]:


fig = plt.figure(figsize = (8, 4))
_ = sns.scatterplot(x = 'PC1', 
                    y = 'PC2', 
                    hue = 'class', 
                   data = dataframe_standerized)


# In[31]:


from mpl_toolkits.mplot3d import Axes3D

crab_colors = {'bluemale':'blue',
               'bluefemale':'yellow',
               'orangemale':'green',
               'orangefemale':'orange'}

fig = plt.figure(figsize = (8, 4))
ax = fig.add_subplot(111, projection = '3d')


plt.scatter(
    dataframe_standerized.PC1,
    dataframe_standerized.PC2,
    zs=dataframe_standerized.PC3,
    depthshade=False,
    c=dataframe['class'].apply(lambda x: crab_colors[x]), 
    s=100
)


# In[ ]:




