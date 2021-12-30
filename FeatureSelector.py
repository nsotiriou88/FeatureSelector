#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Thu Dec 30 14:34:17 2021

@author: Nikolaos Sotiriou
@github: nsotiriou88
@email:  nsotiriou88@gmail.com
'''

# Version. Increase with changes/development (master.dev.minor_changes)
__version__ = '0.1'

# Import libraries
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence
import variance_inflation_fact
import scorecardpy as scpy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_ validate, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.feature_selection import mutual_info_classif, SelectFromModel
from sklearn.preprocessing import StandardScaler, power_transform, MinMaxScaler
from sklearn.base import clone
from joblib import Parallel, delayed

try: 
    from tqdm.notebook import tqdm
except:
    from tqdm import tqdm


##########################################################
#                    Module Functions                    #
########################################################## 
def transformations(X, combine=False): 
    '''
    Applies power transformations to the given features (Box-Cox & Yeo-Johnson).
    
    Parameters
    ----------
    X: pd.DataFrame
    Dataframe containing the features.

    combine: bool, default False
    Determines if it return the transformed features in the same dataframe that
    was passed to the function or as a separate dataframe.

    Returns
    -------
    X_tf: pd.DataSeries
    Transformed features.
    ''' 
    cols = X.columns.to_numpy()
    X_tf = X.copy()

    for col in tqdm(cols, desc='Applying Transformations. Progress: '):
        # remove null features
        if len(X_tf[col].unique())==1:
            X_tf = X_tf.drop(col, axis=1)
            continue
        # convert to Box-Cox or Yea-Johnson
        if (X_tf[col]<0).sum()==0:
            name = col+'_BOX_COX'
            X_tf[name] = power_transform((X_tf[col]+1).values.reshape(-1,1),\
                method='box-cox') 
        else:
            name = col+'YEO_JON'
            X_tf[name] = power_transform(X_tf[col].values.reshape(-1,1),\
                method='yeo-johnson')
        X_tf = X_tf.drop(col, axis=1) 
    
    if combine:
        X_tf = pd.concat([X, X_tf], axis=1)
    
    return X_tf


def f_test(X, y, kbest=None, p_vals=False):
    '''
    F statistics test. This will return p-values and ranks of features based on
    their values.

    Parameters
    ----------
    X: pd.DataFrame
    Dataframe containing the features.

    y: pd.DataFrame or array-like
    Target array.

    kbest: int, default None
    Best number of features to select. If None, it will rank and return all
    given features.
    
    p_vals: bool, default False
    Weather to return original P-values alongside with the ranking score.
    P-values have applied "-np.log10" transformation.
    
    Returns
    -------
    results: pd.DataSeries or tuple (ranks, p_vals)
    Ranked features based on their score performance. If "p_vals" is set to
    True, then it also returns original P-values in a second Series object.
    '''
    if kbest is None:
        f, p = f_classif(X, y)
        scores = -np.log10(p)
        results = pd.Series(scores, index=X.columns, name='f_test_rank')
    else:
        selector = SelectKBest(f_classif, k=kbest)
        selector.fit(X, y)
        scores = -np.log10(selector.pvalues_[selector.get_support()])
        results = pd.Series(scores, index=X.columns[selector.get_support()], name='f_test_rank')
    #check if P-vals to be returned
    if p_vals:
        temp = results.copy()
        temp.name = 'f_test_Pval_negLog10'
        results = (results.rank(ascending=True, method='average',\
            na_option='top').round().astype(np.int), temp)
    else:
        results = results.rank(ascending=True, method='average',\
            na_option='top').round().astype(np.int) 
    
    return results


def chi_test(X, y, minmax_scale=False, kbest=None, p_vals=False):
    '''
    Chi square statistics test for non-negative features. This will return
    P-values and ranks of features based on their values.

    Parameters
    ----------
    X: pd.Dataframe
    Dataframe containing the features.

    y: pd.DataFrame or array-like
    Target array.

    minmax_scale: bool, default False
    Weather to apply MinMax scaler to ensure that all features have non-negative
    values.

    kbest: int, default None
    Best number of features to select. If None, it will rank and return all
    given features.

    p_vals: bool, default False
    Weather to return original P-values alongside with the ranking score.
    P-values have applied "-np.log10" transformation.

    Returns
    -------
    results: pd.DataSeries or tuple (ranks, p_vals)
    Ranked features based on their score performance. If "p_vals" is set to
    True, then it also returns original P-values in a second Series object.
    '''
    # Normalise to avoid negative values
    if minmax_scale:
        X_norm = MinMaxScaler().fit_transform(X)
    else:
        X_norm = X
    if kbest is None:
        chi2_val, p = chi2(X_norm, y)
        scores = -np.log10(p)
        results = pd.Series(scores, index=X.columns, name='chi2_test_rank')
    else:
        selector = SelectKBest(chi2, k=kbest)
        selector.fit(X_norm, y)
        scores = -np.log10(selector.pvalues_[selector.get_support()])
        results = pd.Series(scores, index=X.columns[selector.get_support()],\
            name='chi2_test_rank')
    # check if P-vals to be returned
    if p_vals:
        temp = results.copy()
        temp.name = 'chi2_test_Pval_negLog10'
        results = (results.rank(ascending=True, method='average',\
            na_option='top').round().astype(np.int), temp)
    else:
        results = results.rank(ascending=True, method='average',\
            na_option='top').round().astype(np.int)
    
    return results


def mitest(X, y, n_neighbors=3, mi_vals=False):
    '''
    Mutual information value test.

    Mutual information between two random variables is a non-negative value,
    which measures the dependency between the variables. It is equal to zero if
    and only if two random variables are independent, and higher values mean
    higher dependency.

    The function relies on nonparametric methods based on entropy estimation
    from k-nearest neighbors distances (check
    sklearn.feature_selection.mutual_info_classif).
    
    Parameters
    ----------
    X: pd.DataFrame
    Dataframe containing the features.

    y: pd.DataFrame or array-like
    Target array.

    n_neighbors: int, default 3
    Numbers of neighbours to use for MI estimation of continuous variables.

    mi_vals: bool, default False
    Weather to return original MI values alongside with the ranking score.

    Returns
    -------
    results: pd.DataSeries or tuple (ranks, p_vals)
    Ranked features based on their score performance. If "mi_vals" is set to
    True, then it also returns original MI values in a second Series object.
    '''
    mi = mutual_info_classif(X, y, n_neighbors=n_neighbors)
    results = pd.Series(mi, index=X.columns, name='mi_test_rank')
    # check if mi-vals to be returned
    if mi_vals:
        temp = results.copy()
        temp.name = 'mi_test_val'
        results = (results.rank(ascending=True, method='average',\
            na_option='top').round().astype(np.int), temp) 
    else:
        results = results.rank(ascending=True, method='average',\
            na_option='top').round().astype(np.int)
    
    return results


def iv_test(X, y):
    '''
    Information Value from scorecard model (binned logistic regression).
    
    Parameters
    ----------
    X: pd.DataFrame
    Dataframe containing the features.

    y: pd.DataFrame or array-like
    Target array. Creates the "bad flag" column.

    Returns
    -------
    results: pd.DataSeries
    Ranked features based on their score performance.

    iv_values: pd.DataSeries
    The actual Information Value scores.

    bins: dict of pd.DataFrame objects
    Contains all scorecard model information for each feature:
    bins, WOE, IV, ranges etc.
    '''
    y = pd.DataFrame(y, columns=['bad'])
    # WOE binning
    bins = scpy.woebin(pd.concat([X,y], axis=1), y='bad')
    # Get IV Values
    total_iv = []
    columns = []
    for feat in bins.keys():
        columns.append(feat)
        total_iv.append(bins[feat]['total_iv'].values[0])
    
    results = pd.Series(total_iv, index=columns, name='total_iv_rank')
    iv_values = results.copy()
    iv_values.name = 'total_iv_val'
    results = results.rank(ascending=True, method='average',\
        na_option='top').round().astype(np.int)
    
    return (results, iv_values, bins)


def filter_transformed(scores, identifiers=['_BOX_COX', '_YEO_JON']):
    '''
    After ranking with both normal and transformed features, run this function
    to keep only the top of the two features.
    
    Transformed features should be having some kind of string 
    
    '''





