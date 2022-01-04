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
from statsmodels.stats.outliers_influence import variance_inflation_fact
import scorecardpy as scpy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
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
        results = pd.Series(scores, index=X.columns[selector.get_support()],
                            name='f_test_rank')
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
    
    Parameters
    ----------
    scores: pd.Series
    Pandas DataSeries containing the scores of the features after ranking them.
    
    identifiers: list, default ['_BOX_COX', '_YEO_JON'
    List that contains str identifiers that are attached to the end of the
    transformed features.

    Returns
    -------
    scores_new: pd.DataSeries
    Removed the duplicate transformed features. Transformed features have the
    description removed from their index.
    
    transformed: pd.DataSeries
    Binary Series indicating if the transformed feature was used instead. 1
    indicates transformation was applied.

    original_index: list
    List that contains the original names of the features.
    '''

    scores = scores.sort_values(ascending=False)
    feats = scores.index.to_list()
    feats_new = []
    feats_new_check = []

    while len(feats) > 0:
        check = feats[0]
        for ide in identifiers:
            check = check.replace(ide, '')
        if check not in feats_new_check:
            feats_new_check.append(check)
            feats_new.append(feats[0])
        feats.pop(0)
    
    scores_new = scores[feats_new].copy()
    scores_new = scores_new.rank(ascending=True, method='average',
                                 na_option='top').round().astype(np.int)
    
    # Remove transformation description from index
    new_index = []
    transformed = []
    for val in scores_new.index.values:
        tf = 0
        for ide in identifiers:
            if ide in val:
                val = val.replace(ide, '')
                tf = 1
        new_index.append(val)
        transformed.append(tf)

    # update indices
    original_index = scores_new.index.to_list()
    scores_new.index = new_index
    transformed = pd.Series(transformed, index=new_index, name='tf_mask')
    transformed - transformed.astype(np.int) 

    return (scores_new, transformed, original_index)


def calculate_metrics(scores, cv_results, criterion, clf_choose):
    '''
    This function registers necessary metrics to a given dictionary during the
    recursive factor elimination process. It returns the dictionary.
    '''
    #criterion dictionary
    score_dict = dict(auc='test_roc_auc',
                    roc='test_roc_auc',
                    roc_auc='test_roc_auc',
                    gini='test_roc_auc',
                    accuracy='test_accuracy',
                    recall='test_recall',
                    precision='test_precision',
                    f1='test_f1')
    score_metric = score_dict[criterion]
    
    #accuracy
    scores['best_train']['accuracy'].insert(0, round(100*np.max(
        cv_results['train_accuracy']),2))
    scores['best_test']['accuracy'].insert(0, round(100*np.max(
        cv_results['test_accuracy']),2))
    scores['mean_train']['accuracy'].insert(0, round(100*np.mean(
        cv_results['train_accuracy']),2))
    scores['mean_test']['accuracy'].insert(0, round(100*np.mean(
        cv_results['test_accuracy']),2))
    #roc_auc
    scores['best_train']['roc_auc'].insert(0, round(100*np.max(
        cv_results['train_roc_auc']),2))
    scores['best_test']['roc_auc'].insert(0, round(100*np.max(
        cv_results['test_roc_auc']),2))
    scores['mean_train']['roc_auc'].insert(0, round(100*np.mean(
        cv_results['train_roc_auc']),2))
    scores['mean_test']['roc_auc'].insert(0, round(100*np.mean(
        cv_results['test_roc_auc']),2))
    #recall
    scores['best_train']['recall'].insert(0, round(100*np.max(
        cv_results['train_recall']),2))
    scores['best_test']['recall'].insert(0, round(100*np.max(
        cv_results['test_recall']),2))
    scores['mean_train']['recall'].insert(0, round(100*np.mean(
        cv_results['train_recall']),2))
    scores['mean_test']['recall'].insert(0, round(100*np.mean(
        cv_results['test_recall']),2))
    #precision
    scores['best_train']['precision'].insert(0, round(100*np.max(
        cv_results['train_precision']),2))
    scores['best_test']['precision'].insert(0, round(100*np.max(
        cv_results['test_precision']),2))
    scores['mean_train']['precision'].insert(0, round(100*np.mean(
        cv_results['train_precision']),2))
    scores['mean_test']['precision'].insert(0, round(100*np.mean(
        cv_results['test_precision']),2))
    #f1
    scores['best_train']['f1'].insert(0, round(100*np.max(
        cv_results['train_f1']),2))
    scores['best_test']['f1'].insert(0, round(100*np.max(
        cv_results['test_f1']),2))
    scores['mean_train']['f1'].insert(0, round(100*np.mean(
        cv_results['train_f1']),2))
    scores['mean_test']['f1'].insert(0, round(100*np.mean(
        cv_results['test_f1']),2))
    #gini
    scores['best_train']['gini'].insert(0,
        2*scores['best_train']['roc_auc'][0]-100)
    scores['best_test']['gini'].insert(0,
        2*scores['best_test']['roc_auc'][0]-100)
    scores['mean_train']['gini'].insert(0,
        2*scores['mean_train']['roc_auc'][0]-100)
    scores['mean_test']['gini'].insert(0,
        2*scores['mean_test']['roc_auc'][0]-100)
    #get best estimator based on 'clf_choose'
    if clf_choose=='best':
        scores['best_estimator'].insert(0, cv_results['estimator']\
            [np.argmax(cv_results[score_metric])])
    elif clf_choose=='worst':
        scores['best_estimator'].insert(0, cv_results['estimator']\
            [np.argmin(cv_results[score_metric])])
    else:
        arr = np.array(cv_results[score_metric])
        pos = np.where(arr>=np.median(arr))
        pos = np.min(pos[0])
        scores['best_estimator'].insert(0, cv_results['estimator'][pos])
    
    return scores


def ranking_rfe(feat_stats): 
    '''
    This function returns the scores in ranking format. It also returns the
    'featstats' in DataFrame format.
    '''
    #convert to df the 'feat_stats'
    feat_stats_new = pd.DataFrame(index=feat_stats[0]['features'])
    for ii in sorted(feat_stats.keys()):
        name = 'iter_'+str(len(feat_stats[ii]['values']))
        temp = pd.Series(feat_stats[ii]['values'], index=feat_stats[ii]\
            ['features'], name=name)
        feat_stats_new[name] = np.nan
        feat_stats_new[name] = temp
    #get ranks based on RFE for all features
    results = feat_stats_new.copy()
    results = results.rank(axis=0, ascending=True, method='average',
        na_option='top').round().astype(np.int)
    results = results.sum(axis=1)
    results = results.rank(ascending=True, metbod='average',
        na_option='top').round().astype(np.int) 
    results.name = None

    return results, feat_stats_new


def Recursive_Factor_EliminationCV(estimator, X, y, step=1,
                                   min_features_to_select=1, standardise=False,
                                   criterion='roc_auc', clf_choose='median',
                                   cv=5, n_jobs=1):
    '''
    Eliminates factors based on coefficient value or feature importances.
    Elimination is defined by steps in a recursive manner. It keeps track of all
    performances from cross-validation, during the process.

    Needs an estimator with defined parameters to be passed into the function
    (not fitted).

    Parameters
    ----------
    estimator: sklearn object
    The base estimator from which the transformer is built. This must not be a
    fitted estimator. The estimator must have either the 'feature_importances_'
    or the 'coef_' attribute implemented after fitting.

    X: pd.DataFrame
    Dataframe containing the features.

    y: pd.DataFrame or array-like
    Target array.

    step: int, default 1
    How many features to eliminate in each iteration. 

    min_features_to_select: int, default 1
    The minimum number of features to be selected. This number of features will
    always be scored, even if the difference between the original feature count
    and min_features_to_select isn't divisible by step.

    standardise: bool, default False
    Whether to standardise the input array of features X or not. This can be
    useful if the passed estimator is linear model and the input sample is not
    standardised. It does not affect the original sample given.

    criterion: str, default 'roc_auc'
    Scoring method to obtain the best estimator from cross-validation results on
    each iteration. Passed to 'calculate metrics' function.

    Suported criterion metrics are (validation dataset):
        - 'auc': Area Under the Curve score
        - 'roc': Area Under the Curve score (same with auc)
        - 'roc_auc: Area Under the Curve score (same with auc)
        - 'gini': Gini score (same with auc)
        - 'accuracy': Accuracy score
        - 'recall': Recall score
        - 'precision': Precision score
        - 'f1': F1 score
    
    clf_choose: str, default 'median'
    Method to choose optimal classifier from cross-validation sets. This might
    effect the feature selection on each iteration.

    Suported parameters:
        - 'best': Best performing, based on validation sample
        - 'worst': Worst performing, based on validation sample
        - 'median': Mean/median performing, based on validation sample.
    
    cv: int, default 5
    Uses this number for StratifiedKFold based on target.

    n_jobs: int, default 1
    Number of cores to run in parallel while fitting across folds. Default value
    is to use one thread. For using all threads value is -1.

    Returns
    -------
    results: pd.DataSeries
    Ranked features based on their score performance.

    feat_stats: pd.DataFrame
    All iterations and the coefficients or feature importances of the optimal
    model.

    scores: dict
    All metrics recorded in each iteration. Includes best estimator from each
    iteration. Scores is a nested dictionary of lists in reverse order that the
    RFE was performed (e.g. 'min_features_to_select' is the first in the list
    order).
    '''
    k = 0
    feat_stats = {}
    scores = dict(best_train=dict(accuracy=[], roc_auc=[], recall=[],
                  precision=[], f1=[]),
                  best_test=dict(accuracy=[], roc_auc=[], recall=[],
                  precision=[], f1=[]),
                  mean_train=dict(accuracy=[], roc_auc=[], recall=[],
                  precision=[], f1=[]),
                  mean_test=dict(accuracy=[], roc_auc=[], recall=[],
                  precision=[], f1=[]),
                  best_estimator=[])
    temp_feat = X.columns.to_numpy()
    if standardise:
        scaler_std = StandardScaler()
        X_new = pd.DataFrame(scaler_std.fit_transform(X), columns=X.columns,
            index=X.index)
    else:
        X_new = X
    
    for ii in tqdm(range(X_new.shape[1], min_features_to_select, -step),\
        desc='Progress: '):
        feat_stats[k] = {}
        clf = clone(estimator)
        skf = StratifiedKFold(n_splits=cv, shuffle=True)
        cv_results = cross_validate(clf, X_new[temp_feat], y, n_jobs=n_jobs,
            cv=skf, scoring=['accuracy','roc_auc','recall','precision','f1'],
            return_train_score=True, return_estimator=True)
        #calculate metrics and stats
        scores = calculate_metrics(scores, cv_results, criterion, clf_choose)
        feat_stats[k]['features'] = temp_feat
        if hasattr(scores['best_estimator'][0], 'coef_'):
            feat_stats[k]['values'] = scores['best_estimator'][0].coef_[0]
        elif hasattr(scores['best_estimator'][0], 'feature_importances_'):
            feat_stats[k]['values'] = scores['best_estimator'][0]\
                .feature_importances_
        else:
            print('ERROR: Estimator not supporting "coef_" or "feature_importances_".')
            return
        #remove bottom features
        k += 1
        if (ii-step) > min_features_to_select:
            selector = SelectFromModel(scores['best_estimator'][0],
                threshold=-np.inf, prefit=True, max_features=ii-step)
            temp_feat = temp_feat[selector.get_support()]
        else:
            selector = SelectFromModel(scores['best_estimator'][0],
                threshold=-np.inf, prefit=True,
                max_features=min_features_to_select)
            temp_feat = temp_feat[selector.get_support()]
    # one last cycle for 'min_features_to_select" features
    feat_stats[k] = {}
    clf = clone(estimator)
    skf = StratifiedKFold(n_splits=cv, shuffle=True)
    cv_results = cross_validate(clf, X_new[temp_feat], y, n_jobs=n_jobs, cv=skf,
        scoring=['accuracy','roc_auc','recall','precision','f1'],
        return_train_score=True, return_estimator=True)
    #calculate metrics and stats
    scores = calculate_metrics(scores, cv_results, criterion, clf_choose)
    feat_stats[k]['features'] = temp_feat
    if hasattr(scores['best_estimator'][0], 'coef_'):
        feat_stats[k]['values'] = scores['best_estimator'][0].coef_[0]
    elif hasattr(scores['best_estimator'][0], 'feature_importances_'):
        feat_stats[k]['values'] = scores['best_estimator'][0]\
            .feature_importances_
    else:
        print('ERROR: Estimator not supporting "coef_" or "feature_importances_".')
        return
    #rank and present results in dataframe
    results, feat_stats = ranking_rfe(feat_stats)

    return (results, feat_stats, scores)


def calculate_vif(df, features=None, n_jobs=None):
    '''
    Calculates VIF and returns values in DataFrame format.
    
    Parameters
    ----------
    df: pd.DataFrame
    Dataframe containing the features, without the target column.

    features: list, default None
    List of features to include for the VIF calculation, based on the input
    dataframe. If None, it will calculate for all given features.

    n_jobs: int or -1, default None
    Number of threads to use for VIF calculation. If not set, it will use one
    thread, meaning that it will calculate one VIF at a time for all given
    features.

    Returns
    -------
    vif: pd.DataFrame
    Dataframe that contains final VIF scores in one columns and the features in
    another called 'VAR'.
    '''
    if features is None:
        features = df.columns
    vif = pd.DataFrame()
    vif['VAR'] = features
    if n_jobs is None:
        vif['VIF'] = [variance_inflation_factor(df[features].values, ii) \
            for ii in range(df[features].shape[1])]
    else:
        vif['VIF'] = Parallel(n_jobs=n_jobs)(delayed(variance_inflation_factor)\
            (df[features].values, ii) for ii in range(len(features)))
    
    return vif


def remove_bottom_performers(df, remove=1, vif_col='VIF'):
    '''
    Removes a set ammount of highly correlated features and returns a DataFrame
    with the remaining.

    Parameters
    ----------
    df: pd.DataFrame
    Dataframe containing the VIF scores from 'calculate_vif'.

    remove: int, default 1
    How many bottom performing features to remove.

    vif_col: str, default 'VIF'
    Name of the column that contains the VIF scores. It defaults to 'VIF' as
    given from the 'calculate_vif' function.

    Returns
    -------
    vif: pd.DataFrame
    Dataframe that contains final VIF scores in one columns and the features as
    index of the dataframe.
    '''
    df_sorted = df.sort_values(by=vif_col, ascending=True, ignore_index=False)
    df_sorted = df_sorted.head(df.shape[0]-remove)
    df_sorted = df_sorted.sort_index(ascending=True)

    return df_sorted


def Recursive_Elimination_VIF(df, step=1, threashold=10.0,
                              min_features_to_select=None,
                              features=None, n_jobs=None):
    '''
    Performs recursive factor elimination based on a defined step. Target is to
    either reduce features size to a specified number, or a threshold that all
    remaining features are lower than that.

    The method uses OLS to predict one feature with the rest of the remaining
    features and this is how the VIF score is calculated. Non-linear correlated
    features have small VIF score.

    Parameters
    ----------
    df: pd.DataFrame
    Dataframe containing the features, without the target column.

    step: int, default 1
    How many features to eliminate in each iteration. Based on 'threashold' and
    'min_features_to_select' parameters, it may be that not all features are
    removed in this step.

    threashold: float, default 10.0
    VIF threashold to stop removing features after this point.

    min_features_to_select: int, default 1
    The minimum number of features to be selected. If this set, then
    'threashold' is ignored.

    features: list, default None
    List of features to include for the VIF calculation, based on the input
    dataframe. If None, it will calculate for all given features.

    n_jobs: int or -1, default None
    Number of threads to use for VIF calculation. If not set, it will use one
    thread, meaning that it will calculate one VIF at a time for all given
    features.

    Returns
    -------
    vif: pd.DataFrame
    Dataframe that contains final VIF scores in one columns and the features in
    another called 'VAR'.
    
    drop_order: pd.DataFrame
    Which feature was dropped in each iteration. Position 0 is the first dropped
    feature.
    '''
    if features is None:
        features = df.columns
    if min_features_to_select is None:
        min_features_to_select = 1
    else:
        threashold = None
    drop_list = []

    for ii in tqdm(range(df.shape[1]-step, min_features_to_select, -step),
        desc='Progress: '):
        vif = calculate_vif(df, features=features, n_jobs=n_jobs)
        #check removing factor(s)
        to_drop = vif.sort_values(by='VIF', ascending=False,
            ignore_index=False).head(step)
        if threashold is not None: #removing based on 'threashold'
            if to_drop['VIF'].min()>=threashold:
                for var in to_drop['VAR'].values:
                    drop_list.append(var)
                vif = remove_bottom_performers(vif, remove=step)
            else:
                break
        else: #remove based on 'min_features_to_select'
            if len(vif)-len(to_drop)<min_features_to_select:
                break
            else:
                for var in to_drop['VAR'].values:
                    drop_list.append(var)
                vif = remove_bottom_performers(vif, remove=step)
        features = vif['VAR'].values #new list of features for next iter
    #constract dataframe based on order of elimination
    drop_order = {ii: var for ii, var in enumerate(drop_list)}
    jj = len(drop_order)
    for var in vif.sort_values(by='VIF', ascending=False)['VAR'].values:
        drop_order[jj] = var
        jj += 1
    drop_order = pd.DataFrame.from_dict(drop_order, orient='index',
                                        columns=['VAR'])
    
    return (vif, drop_order)


def Forward_Elimination_UIF(df, scores, topk=None, score_col=None, step=1,
                            threashold=10.0, verbose=False, n_jobs=None):
    '''
    Performs recursive factor elimination in a forward manner. It removes
    factors above the defined threashold, until the top K features based on a
    score column, are all having VIF of less than the specified threshold. In
    each iteration, the VIF for the "topk" features is calculated, removes
    "step" features if above defined threashold and replaces them with the next
    features in the order. In every iteration, the VIF for "topk" number of
    features is always calculated. The process stops if there are no more
    features to replace next or the VIF factors of all "topk" features are less
    than "threashold". It is faster than starting with the total pool of
    features and eliminating backwards, which also can remove important features
    in the process.
    
    The method uses OLS to predict one feature with the rest of the remaining
    features and this is how the VIF score is calculated. Non-linear correlated
    features have small VIP score.
    
    Parameters
    ----------
    df: pd.DataFrame
    Dataframe containing the features, without the target column. If 'score_col'
    is not defined, this df must be sorted based on the best features on top
    (descending score order).

    scores: pd.DataFrame or pd-Series
    Pandas dataset that contains the features to be tested as index and also
    contains the scoring column.

    topk: int, None
    This determines how many top K features we desire. It is also the starting
    number of features to start calculating VIF and removing features based on
    'threashold' and 'step'. If None, it will get top 25%, eg.
    int(len(scores)/4).

    score_col: str, default None
    Name of the column that keeps the scoring of features. If not defined, it
    will assume that the input 'scores' is sorted with descending feature
    importance (index).

    step: int, default 1
    How many features to eliminate in each iteration. Based on 'threashold'
    parameter, it removes features from the VIF calculation and adds the next
    'step' ones (if available).

    threashold: float, default 10.0
    VIF threashold to stop removing features after this point.

    verbose: bool, default False
    Prints results from every iteration of which features have been eliminated,
    their VIF and the top 10% VIF. This helps to understand when the algorithm
    is close to finalise the list of "topk" best non-linearly correlated
    features.

    n_jobs: int or -1, default
    None Number of threads to use for VIF calculation. If not set, it will use
    one thread, meaning that it will calculate one VIP at a time for all given
    features.

    Returns
    -------
    vif: pd.DataFrame
    Dataframe that contains final VIF scores in one columns and the features in
    another called 'VAR'. Only 'topk' features would have their VIF displayed.

    drop_order: pd.DataFrame
    Which feature was dropped in each iteration. Position 0 is the first dropped
    feature.
    '''
    if topk is None:
        topk = int(len(scores)/4)
    if score_col is None:
        features = scores.index.values[:topk]
        features_to_test = scores.index.values[topk:]
    else:
        features = scores.sort_values(score_col, ascending=False)\
            .index.values[:topk]
        features_to_test = scores.sort_values(score_col, ascending=False).index\
            .values[topk:]
    drop_list = []

    for ii in tqdm(range(scores.shape[0]-topk-step, 0, -step),
        desc='Progress: '):
        vif = calculate_vif(df, features=features, n_jobs=n_jobs)
        #check removing factor(s)
        to_drop = vif.sort_values(by='VIF', ascending=False,
            ignore_index=False).head(step)
        if to_drop['VIF'].min()>=threashold:
            for var in to_drop['VAR'].values:
                drop_list.append(var)
            vif = remove_bottom_performers(vif, remove=step)
        else:
            break
        #print diagnostics
        if verbose:
            print('-------------')
            print('Dropping features with VIF(threashold '+str(threashold)+'):')
            print(to_drop)
            avg_top10=vif['VIF'][vif['VIF']>=vif['VIF'].quantile(q=0.9)].mean()
            print('Top 10% VIF average:', avg_top10)
            print('-------------')
        #new list of features for next iter
        features = vif['VAR'].values
        features = np.concatenate((features, features_to_test[:step]))
        features_to_test = features_to_test[step:]

    #constract dataframe based on order of elimination
    drop_order = {ii: var for ii, var in enumerate(drop_list)}
    drop_order = pd.DataFrame.from_dict(drop_order, orient='index',
        columns=['VAR'])
    
    return (vif, drop_order)


def plot_metrics_check(cv_scores, sharey=False, savefig=None):
    '''
    This function plots the metrics accumulated over the iterations of feature
    selection processes.

    Parameters
    ----------
    cv scores: dict
    Cross-validation results in the dictionary provided by
    'RecursiveFactorEliminationCV'.

    sharey: bool, default False
    Same Y-axis limits across sub-plots. It can help with visualisation.

    savefig: str, default None
    Weather to save figure or not. It should be the full path+name of the
    location to be saved. eg. '/repos/figure3.png' or just 'figure3.png' to save
    it in the current folder under the given name.
    '''
    sns.set_style('darkgrid')
    fig, axs = plt.subplots(3, 2, sharex=True, sharey=sharey)
    fig.set_size_inches(16,14)
    fig.suptitle('Number of features and Scoring Methods')
    metrics = ['accuracy', 'roc_auc', 'recall', 'precision', 'f1', 'gini']
    plot_confing = dict(
        best_train=dict(linestyle='-', marker='.', color='orange'),
        mean_train=dict(linestyle='--', marker='.', color='orange'),
        best_test=dict(linestyle='-', marker='.', color='green'),
        mean_test=dict(linestyle='--', marker='.', color='green'))
    
    if hasattr(cv_scores['best_estimator'][0], 'coef_'):
        lens = [len(clf.coef_[0]) for clf in cv_scores['best_estimator']]
    elif hasattr(cv_scores['best_estimator'][0], 'feature_importances_'):
        lens = [len(clf.feature_importances_) for clf in cv_scores['best_estimator']]
    else:
        print('ERROR: Estimator not supporting "coef_" or "feature_importances_".') 
        plt.show() #return broken figure
        return
    i, j = 0, 0
    for metric in metrics:
        for score, item in cv_scores.items():
            if score=='best_estimator':
                continue
            axs[i,j].plot(lens, item[metric], label=score,
                          **plot_confing['score'])
        axs[i,j].set_title(metric.upper())
        # increase grid i,j parameters j +- 1
        j += 1
        if j ==2:
            i += 1
            j = 0
    plt.legend(loc='lower right')
    if savefig is not None:
        plt.savefig(savefig, dpi=320)
    plt.show()

    return


def plot_feature_performance(feat_stats, topk=None, iteration=None,
                             legend=False, savefig=None):
    '''
    This function plots the metrics accumulated over the iterations of feature
    seleciton processes.

    Parameters
    ----------
    feat_stats: pd.DataFrame
    Dataframe that contains the weights/importances from each step during the
    'Recursive_Factor_EliminationCV'.

    topk: int, default None
    Number of top performing features to plot, based on the set 'iteration'. 0
    is first iteration with all given features, I is for the second etc. If
    'iteration' is None, it will get the topk features from the last iteration.
    
    iteration: int, default None
    Select which iteration to consider for the topk features. If not set, it
    will get the last iteration. If 'topk' is not set, this parameter is
    ignored.

    legend: bool, default False
    Control if legend will be visible. Can help in case of many variables if not
    needed.

    savefig: str, default None
    Weather to save figure or not. It should be the full path+name of the
    location to be saved. eg. '/repos/figure3.png' or just 'figure3.png' to save
    it in the current folder under the given name.
    '''
    sns.set_style('white')
    fig, ax = plt.subplots(1, 1)
    fig. set_size_inches(16,9)
    df = feat_stats.copy() # not changing original
    cols = df.columns.to_list()
    for ii in range(len(cols)):
        cols[ii] = int(cols[ii].replace('iter_',''))
    df.columns = cols
    if topk is not None:
        if iteration is None:
            iteration = -1
        topk_feats = df.iloc[:,iteration].abs().sort_values(ascending=False)
        topk_feats = topk_feats.head(topk).index.values
        df = df.loc[topk_feats,:]
    df = df.T
    df.plot(kind='line', ax=ax, legend=legend, use_index=True,
        title='Feature Importance vs Number of Features')
    ax.invert_xaxis()
    if savefig is not None:
        plt.savefig(savefig, dpi=320)
    plt.show()

    return



def plot_cv_scores(cv_results, param, scoring, xlim=None, ylim=None, grid=False,
                   loc='best', figsize=(16, 9)):
    '''
    This function plots scores from cross validation results.

    This function can plot the results of 'GridSearchCV.cv_results_', when used
    to validate ONE parameter only. For example, in a Tree-based:

    parameters = {
        'n_estimators': range(100, 2501, 100),
        'criterion': ['gini'],
        'max_depth': [8],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'max_ features': [10],
        'class_ weight': [{0:1, 1:round(ratio)}],
        'n_jobs': [-1],
        'max_samples': [0.85]
    }

    NOTE that we tested only 'n_estimators' in this example.

    Parameters
    ----------
    cv_results: dict
    Dictionary obtained after the '.fit()' method on 'GridSearchCV' object by
    calling the attribute '.cv_results_'. NOTE that you must set parameter
    'return_train_score' to 'True' in order to get the output of the train score
    and compare it.

    param: str
    The parameter that was used to validate through 'GridSearchCV'. It is used
    on the x-axis.

    scoring: str or list
    List of scoring methods that have been used during the 'GridSearchCV', eg.
    ['accuracy', 'f1']. All scores are displayed on y-axis.

    xlim, ylim: tuple, default None
    Control the limits of x-axis and y-axis. If None, matplotlib will use
    default.

    grid: bool, default False
    Use grid in plot.

    loc: str or None, default 'best'
    Define position of legend location. if None, legend will not be diplayed.

    figsize: tuple, default: (16, 9)
    Matplotlib figure size.
    '''
    if type(scoring)!=list:
        scoring = [scoring]
    plt.figure(figsize=figsize)
    plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
              fontsize=16)
    plt.xlabel(param)
    plt.ylabel("Score")
    ax = plt.gca()
    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
        y_mark = ylim[0]
    else:
        y_mark = 0
    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(cv_results['param_'+param].data, dtype=float)
    for scorer, color in zip(sorted(scoring), ['g', 'k', 'r', 'b', 'm', 'y']):
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = cv_results['mean_%s_%s' % (sample, scorer)]
            sample_score_std = cv_results['std_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, sample_score_mean-sample_score_std,
                            sample_score_mean+sample_score_std,
                            alpha=0.1 if sample=='test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                    alpha=1 if sample=='test' else 0.7,
                    label="%s (%s)" % (scorer, sample))
        best_index = np.nonzero(cv_results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = cv_results['mean_test_%s' % scorer][best_index]
        # Plot a dotted vertical line at the best score for that scorer
        # marked by x.
        ax.plot([X_axis[best_index], ] * 2, [y_mark, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3,
                ms=8)
        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score,
                    (X_axis[best_index], best_score + 0.005))
    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
    plt.legend(loc=loc)
    plt.grid(grid)
    plt.show()

    return


def plot_feature_comparison(scores, norm=False, label=None, title=None,
                            figsize=(17, 8), savefig=None):
    '''
    This function plots a bar graph the feature comparison based on the provided
    metric value or rank score.

    Parameters
    ----------
    scores: pd.Series or array-like
    Score values array, based on rank score or metric.
    
    norm: bool, default False
    If values need to be normalised based on max value.

    label: str, default None
    Label to be used for the scoring graph.

    title: str, default None
    Title to be used for the scoring graph.

    figsize: tuple, default: (17, 8)
    Matplotlib figure size.

    savefig: str, default None
    Weather to save figure or not. It should be the full path+name of the
    location to be saved. eg. '/repos/figure3.png' or just 'figure3.png' to save
    it in the current folder under the given name.
    '''
    if norm:
        scores /= scores.max()
    if label is None:
        label = 'Score Values'
    if title is None:
        title = 'Comparing feature selection'
    
    sns.set_style('darkgrid')
    plt.figure(figsize=figsize)
    plt.bar(np.arange(1, len(scores)+1), scores, width=0.3, label=label)
    plt.title(title)
    plt.xlabel('Feature number')
    plt.axis('tight')
    plt.legend(loc='upper right')
    
    if savefig is not None:
        plt.savefig(savefig, dpi=320)
    plt.show()

    return

