#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_digits
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[3]:


X, y = load_digits(return_X_y = True)


# In[9]:


plt.figure(figsize =(3, 3) )
plt.imshow(X[2].reshape(8, 8), cmap = 'Greys')


# In[12]:


fig, axes = plt.subplots(nrows = 2, ncols = 3, sharex = True, sharey = True, figsize = (6, 6))

axes[0, 0].imshow(X[2].reshape(8, 8), cmap = 'Greys')
axes[0, 1].imshow(X[1].reshape(8, 8), cmap = 'Greys')
axes[0, 2].imshow(X[2].reshape(8, 8), cmap = 'Greys')
axes[1, 0].imshow(X[3].reshape(8, 8), cmap = 'Greys')
axes[1, 1].imshow(X[4].reshape(8, 8), cmap = 'Greys')
axes[1, 2].imshow(X[5].reshape(8, 8), cmap = 'Greys')

plt.tight_layout()


# ## Fisher Discriminant Analysis :

# Fisher Discriminant Analysis (FDA), also known as Linear Discriminant Analysis (LDA), is a statistical method used for dimensionality reduction and classification. It aims to find a linear combination of features that best separates two or more classes in a dataset. The objective is to maximize the distance between the means of different classes while minimizing the spread within each class. In essence, FDA seeks to identify the features that contribute the most to the differences between classes, making it a useful technique in pattern recognition and machine learning for classification tasks.

# ### LDA Analysis on 2 Dimensions :

# In[14]:


lda = LinearDiscriminantAnalysis(n_components = 2)

lda_x_dataframe = lda.fit_transform(X, y = y)

dataframe = pd.DataFrame({'FDA1': lda_x_dataframe[:, 0], 'FDA2': lda_x_dataframe[:, 1], 'class': y})


# In[15]:


dataframe.head()


# In[17]:


fig = plt.figure(figsize = (8, 5))
sns.scatterplot(x = dataframe.FDA1, 
                y = dataframe.FDA2, 
                hue = 'class', 
                data = dataframe, 
                palette = 'tab10')


# ### LDA Analysis on 3 Dimensions :

# In[21]:


lda = LinearDiscriminantAnalysis(n_components = 3)

lda_x_dataframe = lda.fit_transform(X, y = y)

dataframe = pd.DataFrame({'FDA1': lda_x_dataframe[:, 0], 
                          'FDA2': lda_x_dataframe[:, 1], 
                          'FDA3': lda_x_dataframe[:, 2], 
                          'class': y})


# In[22]:


dataframe.head()


# In[25]:


fig = plt.figure(figsize=(8, 4))

ax = fig.add_subplot(111, projection='3d')

plt.scatter(x=dataframe.FDA1,
            y=dataframe.FDA2,
            zs=dataframe.FDA3,
            c=dataframe['class'], s=100)


# In[ ]:




