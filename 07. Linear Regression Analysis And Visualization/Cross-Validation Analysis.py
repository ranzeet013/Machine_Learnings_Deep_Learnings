#!/usr/bin/env python
# coding: utf-8

# Conducting k-fold cross-validation with k=5, which trains a Linear Regression model on different subsets of the data. It evaluates the model's performance using metrics like Mean Squared Error (MSE), normalized MSE (NMSE), and R-squared (R2) for each fold. The results are systematically recorded in a DataFrame, `cross_val_metrics`, facilitating a thorough analysis of the model's consistency and generalization across diverse data splits.

# In[3]:


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


# In[4]:


cross_val_metrics.loc['Mean',:] = cross_val_metrics.mean()

cross_val_metrics


# In[ ]:




