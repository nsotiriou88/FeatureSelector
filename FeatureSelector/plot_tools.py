#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Wed Jan 5 11:18:19 2022

@author: Nikolaos Sotiriou
@github: nsotiriou88
@email:  nsotiriou88@gmail.com
'''

# Version. Increase with changes/development (master.dev.minor_changes)
__version__ = '0.3'

# Import libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


##########################################################
#               Module Plotting Functions                #
##########################################################
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
    scores = scores.copy()
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
    plot_config = dict(
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
            axs[i,j].plot(lens, item[metric],label=score,**plot_config[score])
        axs[i,j].set_title(metric.upper())
        # increase grid i,j parameters
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
                             from_iteration=None, to_iteration=None,
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
    Select which iteration to consider for considering the topk features. If not
    set, it will get the last iteration. If 'topk' is not set, this parameter is
    ignored.
    
    from_iteration: int, default None
    Select which iteration to consider for start plotting. If not set, it will
    start from the first iteration.
    
    to_iteration: int, default None
    Select which iteration to consider for end plotting. If not set, it will
    start from the first iteration.

    legend: bool, default False
    Control if legend will be visible. Can help in case of many variables if not
    needed.

    savefig: str, default None
    Weather to save figure or not. It should be the full path+name of the
    location to be saved. eg. '/repos/figure3.png' or just 'figure3.png' to save
    it in the current folder under the given name.
    '''
    df = feat_stats.copy() # not changing original
    #check start
    if from_iteration is None:
        from_iteration = 0
    
    sns.set_style('white')
    fig, ax = plt.subplots(1, 1)
    fig. set_size_inches(16,9)
    cols = df.columns.to_list()
    #remove 'iter_' from column name
    for ii in range(len(cols)):
        cols[ii] = int(cols[ii].replace('iter_',''))
    df.columns = cols
    if topk is not None:
        if iteration is None:
            iteration = -1
        topk_feats = df.iloc[:,iteration].abs().sort_values(ascending=False)
        topk_feats = topk_feats.head(topk).index.values
        df = df.loc[topk_feats,:]
    #transpose for plotting with pandas
    df = df.T
    #filter end and start
    if to_iteration is None:
        df = df.iloc[from_iteration:, :]
    else:
        df = df.iloc[from_iteration:to_iteration, :]
    iters = df.shape[0] #get total iterations
    df.plot(kind='line', style='.-', ax=ax, legend=legend, use_index=True,
        title=f'Feature Importance vs Number of Features Remaining ({iters} iterations)')
    ax.invert_xaxis()
    plt.ylabel('Feature importance')
    plt.xlabel('Remaining features')
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

