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

from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from statsmodels.genmod.generalized_linear_model import GLM
import statsmodels.api as sm

from scipy import stats


from yellowbrick.regressor import residuals_plot
from yellowbrick.regressor import prediction_error


# In[2]:


dataframe = pd.read_csv('LifeExpectancy.csv', index_col = [0])


# In[3]:


dataframe.head()


# In[4]:


dataframe.tail()


# In[5]:


dataframe.shape


# In[6]:


dataframe.info()


# ### Preprocessing :

# Preprocessing is the steps taken to clean, organize, and transform raw data into a format suitable for analysis or machine learning models. It is a crucial step in the machine learning pipelines, as the quality of the input data directly impacts the performance of the models.

# In[7]:


dataframe.isna().sum().sort_values(ascending = False)


# In[8]:


categorical_columns = list(dataframe.dtypes[dataframe.dtypes == 'O'].index.values)

for column in categorical_columns:
  dataframe[column] = dataframe[column].astype('category')


# In[9]:


X = dataframe.loc[:,dataframe.columns != 'life_expectancy']
y = dataframe['life_expectancy']


# In[10]:


X.head()


# In[11]:


y.head()


# In[12]:


X.shape, y.shape


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X,y, 
                                                    test_size = 0.33, 
                                                    random_state = 42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape


# 
# 
# 1. **Data Cleaning:**
#    - Handling missing data: Imputing missing values or removing rows/columns with missing data.
#    - Removing duplicates: Identifying and removing identical rows from the dataset.
# 
#     - Exploratory data analysis (EDA) through visualization to understand the distribution and relationships within the dataset.
# 

# 
# 
# 
# 2. **Data Splitting:**
#    - Splitting the dataset into training and testing sets for model evaluation.
#    - In some cases, creating validation sets for hyperparameter tuning during model training.
# 
# 

# In[14]:


def cleaning_dataframe(X,y):
  print('Original Size:{}'.format(X.shape))
  categorical_columns = X.dtypes[X.dtypes == 'category'].index.values
  X = X.drop(columns=categorical_columns)                             # removing categorical columns
  print('Removed: {}'.format(categorical_columns))
  X = X.dropna()                                                      # dropping missing values
  y = y[X.index]  
  print('New Size: {}'.format(X.shape))
  return X,y


# In[15]:


X_train, y_train = cleaning_dataframe(X_train,y_train)
X_test, y_test = cleaning_dataframe(X_test,y_test)


# In[16]:


lr = LinearRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_train)


# In[17]:


weights = lr.coef_
intercept = lr.intercept_

print('Coefficients: \n',weights[:25])
print('Interceptor: \n',intercept)


# ### Statmodel Linear Regression :

# In[18]:


model = sm.OLS(y_train, sm.add_constant(X_train))
result = model.fit()
result.summary()


# ### Model Analysis :

# In[19]:


fig, ax = plt.subplots(figsize = (8,4))
ax.set_xlim([-15,15])
sns.distplot(result.resid,bins = 30)


# ### Probability Plot :

# In[20]:


fig, ax = plt.subplots(figsize = (8,4))
stats.probplot(result.resid,plot = plt)


# ### Q-Q Plot :

# The quantile-quantile (q-q) plot is a graphical technique for determining if two data sets come from populations with a common distribution. A q-q plot is a plot of the quantiles of the first dataset against the quantiles of the second dataset.

# In[21]:


plt.figure(figsize=(10, 5));
viz = residuals_plot(lr, 
                     X_train, 
                     y_train, 
                     X_test, 
                     y_test, 
                     is_fitted = True, qqplot = True, hist = False)


# ### Visualizing Prediction W.R.T Actual Values 

# In[22]:


plt.figure(figsize = (8,4))
visualizer = prediction_error(lr, 
                              X_test, 
                              y_test, 
                              is_fitted = True)


# In[23]:


from sklearn.metrics import mean_squared_error
N = X_train.shape[0]

prediction = result.predict(sm.add_constant(X_train))
mean_square_error = np.sum((y_train-prediction)**2)/N

mean_square_error_sk = mean_squared_error(y_train,prediction)

mean_square_error, mean_square_error_sk


# In[24]:


norm_mse = np.sum((y_train - prediction)**2)/((N)*np.var(y_train))
norm_mse


# In[26]:


R_squared_sk = r2_score(y_train,prediction)
R_squared_sk


# In[ ]:




