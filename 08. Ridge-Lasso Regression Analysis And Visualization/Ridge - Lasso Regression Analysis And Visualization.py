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
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import r2_score

from statsmodels.genmod.generalized_linear_model import GLM
import statsmodels.api as sm

from scipy import stats


from yellowbrick.regressor import residuals_plot
from yellowbrick.regressor import prediction_error
from yellowbrick.regressor import AlphaSelection


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


plt.figure(figsize=(10, 5))
viz = residuals_plot(lr, 
                     X_train, 
                     y_train, 
                     X_test, 
                     y_test, 
                     is_fitted=True, qqplot=True, hist=False)

plt.savefig('QQ Plot.png')


# ### Visualizing Prediction W.R.T Actual Values 

# In[22]:


plt.figure(figsize = (8,4))
visualizer = prediction_error(lr, 
                              X_test, 
                              y_test, 
                              is_fitted = True)
plt.savefig('p-error.png')


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


# In[25]:


R_squared_sk = r2_score(y_train,prediction)
R_squared_sk


# ###  Cross-Validation :

# In[26]:


cross_val_metrics = pd.DataFrame(columns=['MSE', 'norm_MSE', 'R2'])

kf = KFold(n_splits=5)
i=1
for train_index, test_index in kf.split(X_train):
    print('Split {}: \n\tTest Folds: [{}] \n\tTrain Folds {}'.format(i, i, [j for j in range(1,6) if j != i]));
    
    x_train_fold = X_train.values[train_index]
    y_train_fold = y_train.values[train_index]
    x_test_fold = X_train.values[test_index,:]
    y_test_fold = y_train.values[test_index]
    
    lr = LinearRegression().fit(x_train_fold,y_train_fold)
    y_pred_fold = lr.predict(x_test_fold)
    fold_mse =mean_squared_error(y_test_fold, y_pred_fold)
    fold_nmse =  1-r2_score(y_test_fold, y_pred_fold)
    fold_r2 = r2_score(y_test_fold, y_pred_fold)
    print(f'\tMSE: {fold_mse:3.3f} NMSE: {fold_nmse:3.3f} R2: {fold_r2:3.3f}')

    cross_val_metrics.loc[f'Fold {i}', :] = [fold_mse,fold_nmse, fold_r2]
    i = i + 1


# In[27]:


cross_val_metrics.loc['Mean',:] = cross_val_metrics.mean()

cross_val_metrics


# ### Ridge Regression :

# In[28]:


ridge_cross_val_metrics = pd.DataFrame(columns=['mean MSE','mean norm_MSE','mean R2'])
lambdas = [1e-4,1e-3,1e-2,0.1,0.5,1,10,50,100]
for lambda_val in lambdas:
  kf = KFold(n_splits=5)
  cv_mse = []
  cv_nmse = []
  cv_r2 = []

  for train_index, test_index in kf.split(X_train):
    x_train_fold = X_train.values[train_index]
    y_train_fold = y_train.values[train_index]
    x_test_fold = X_train.values[test_index,:]
    y_test_fold = y_train.values[test_index]

    lr = Ridge(alpha=lambda_val).fit(x_train_fold,y_train_fold)
    y_pred_fold = lr.predict(x_test_fold)
    fold_mse = mean_squared_error(y_test_fold,y_pred_fold)
    fold_nmse = 1 - r2_score(y_test_fold,y_pred_fold)
    fold_r2 = r2_score(y_test_fold,y_pred_fold)

    cv_mse.append(fold_mse)
    cv_nmse.append(fold_nmse)
    cv_r2.append(fold_r2)
  ridge_cross_val_metrics.loc[f'Lambda={lambda_val}',:] = [np.mean(cv_mse),np.mean(cv_nmse),np.mean(cv_r2)]

ridge_cross_val_metrics.sort_values(by='mean R2',ascending=False)


# In[29]:


ridge_cv = RidgeCV(alphas=lambdas,cv=5).fit(X_train,y_train)

print(f'Best Lambda: {ridge_cv.alpha_} R2 score: {ridge_cv.best_score_:3.4f}')


# In[30]:


plt.figure(figsize=(5, 3))
visualization = AlphaSelection(RidgeCV(alphas=lambdas))
visualization.fit(X_train,y_train)


# ### Lasso Regression :

# In[31]:


lasso_cross_val_metrics = pd.DataFrame(columns=['mean MSE','mean norm_MSE','mean R2'])
lambdas = [1e-4,1e-3,1e-2,0.1,0.5,1,10,50,100]
for lambda_val in lambdas:
  kf = KFold(n_splits=5)
  cv_mse = []
  cv_nmse = []
  cv_r2 = []
  for train_index, test_index in kf.split(X_train):
    x_train_fold = X_train.values[train_index]
    y_train_fold = y_train.values[train_index]
    x_test_fold = X_train.values[test_index,:]
    y_test_fold = y_train.values[test_index]

    lr = Lasso(alpha=lambda_val).fit(x_train_fold,y_train_fold)
    y_pred_fold = lr.predict(x_test_fold)
    fold_mse = mean_squared_error(y_test_fold,y_pred_fold)
    fold_nmse = 1 - r2_score(y_test_fold,y_pred_fold)
    fold_r2 = r2_score(y_test_fold,y_pred_fold)

    cv_mse.append(fold_mse)
    cv_nmse.append(fold_nmse)
    cv_r2.append(fold_r2)
  lasso_cross_val_metrics.loc[f'Lambda={lambda_val}',:] = [np.mean(cv_mse),np.mean(cv_nmse),np.mean(cv_r2)]

lasso_cross_val_metrics.sort_values(by='mean R2',ascending=False)


# In[36]:


plt.figure(figsize=(5, 3))
visualization = AlphaSelection(LassoCV(alphas=lambdas))
visualization.fit(X_train,y_train)


# In[ ]:




