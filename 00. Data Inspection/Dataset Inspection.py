#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries :

# These are popular data science and visualization libraries, including pandas, numpy, seaborn, and matplotlib. Here's a brief description of each library's role:
# 
# 1. **pandas (`import pandas as pd`):** Pandas is a powerful data manipulation and analysis library in Python. It provides data structures like DataFrame for efficient data handling and manipulation.
# 
# 2. **numpy (`import numpy as np`):** NumPy is a numerical computing library for Python. It provides support for large, multi-dimensional arrays and matrices, along with mathematical functions to operate on these elements.
# 
# 3. **seaborn (`import seaborn as sns`):** Seaborn is a statistical data visualization library built on top of matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
# 
# 4. **matplotlib (`import matplotlib.pyplot as plt`):** Matplotlib is a versatile plotting library for Python. It allows users to create a wide variety of static, animated, and interactive plots.
# 
# The line `pd.set_option('display.precision', 3)` sets the display precision for floating-point numbers in pandas DataFrames to three decimal places.

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.precision', 3)


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


dataframe = pd.read_csv('crabs.csv')


# In[4]:


dataframe.head()


# In[5]:


dataframe.tail()


# Changing the columns names for better understanding. The columns are 
# 
# - The mapping is as follows:
#   - 'sp' is renamed to 'species'
#   - 'FL' is renamed to 'frontal_lobe'
#   - 'RW' is renamed to 'rear_width'
#   - 'CL' is renamed to 'carapace_midline'
#   - 'CW' is renamed to 'maximum_width'
#   - 'BD' is renamed to 'body_depth'

# In[6]:


dataframe = dataframe.rename(columns = {'sp': 'specise', 
                                        'FL':'frontal_lobe', 
                                        'RW':'rear_width', 
                                        'CL':'carapace_midline', 
                                        'CW':'maximum_width', 
                                        'BD':'body_depth'})


# In[7]:


dataframe.columns


# 
# 
# 1. `dataframe['species'] = dataframe['species'].map({'B': 'blue', 'O': 'orange'})`: This line maps values in the 'species' column. It replaces 'B' with 'blue' and 'O' with 'orange'. This kind of mapping is useful to convert categorical values to more understandable labels.
# 
# 2. `dataframe['sex'] = dataframe['sex'].map({'F': 'female', 'M': 'male'})`: This line maps values in the 'sex' column. It replaces 'F' with 'female' and 'M' with 'male'.

# In[8]:


dataframe['specise'] = dataframe['specise'].map({'B':'blue',
                          'O':'orange'})
dataframe['sex'] = dataframe['sex'].map({'F':'female', 
                      'M':'male'})


# In[9]:


dataframe.head()


# In[10]:


dataframe.describe(include = 'all')


# 
# 
# - `dataframe['species'] + dataframe['sex']`: This operation concatenates the values from the 'species' column and the 'sex' column element-wise.
# 
# - `dataframe['class'] = ...`: The result of the concatenation is assigned to a new column called 'class' in the DataFrame.
# 

# In[11]:


dataframe['class'] = dataframe.specise + dataframe.sex


# In[12]:


dataframe.head()


# In[13]:


dataframe['class'].value_counts()


# In[17]:


dataframe_columns = ['frontal_lobe','rear_width','carapace_midline','maximum_width','body_depth']
dataframe[dataframe_columns].describe()


# In[23]:


fig, ax = plt.subplots(figsize = (8, 3))

dataframe[dataframe_columns].boxplot()


# ### Visualizing the classes:

# 
# Visualizing classes in a dataset typically involves creating plots or charts to understand the distribution or relationships between different classes. 

# In[25]:


dataframe.boxplot(column='frontal_lobe', 
                   by = 'class', 
                   figsize = (8,5))


# In[26]:


dataframe.boxplot(column = 'rear_width', by = 'class', figsize=(8,5))


# In[28]:


dataframe.boxplot(column='carapace_midline', by = 'class', figsize=(8,5))


# In[29]:


dataframe.boxplot(column='body_depth', by = 'class', figsize=(8,5))


# Histograms are commonly used in data exploration and analysis to understand the central tendency, spread, and overall pattern of a dataset. They are a fundamental tool in descriptive statistics and data visualization

# In[31]:


dataframe[dataframe_columns].hist(figsize=(16,4),layout=(1,6))


# In[32]:


plt.figure(figsize=(8,6))

sns.histplot(dataframe , 
             x = 'frontal_lobe', 
             hue='class', 
             bins=20)


# In[33]:


plt.figure(figsize=(8,6))

sns.histplot(dataframe, 
             x =  'rear_width', 
             hue='class', 
             bins = 20)


# In[34]:


plt.figure(figsize=(8,6))

sns.histplot(dataframe, 
             x = 'carapace_midline', 
             hue = 'class',
             bins = 20)


# In[35]:


plt.figure(figsize=(8,6))

sns.histplot(dataframe, 
             x = 'maximum_width', 
             hue = 'class', 
             bins = 20)


# In[36]:


plt.figure(figsize=(8,6))

sns.histplot(dataframe, 
             x = 'body_depth', 
             hue = 'class', 
             bins=20)


# Pairplots are particularly useful for exploring relationships and identifying patterns in multivariate datasets, providing a quick visual summary of the interactions between variables. 

# In[37]:


sns.pairplot(dataframe, 
             hue = 'class')


# In[ ]:




