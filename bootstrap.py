"""
Bootstrap for significant pixels
- primary loop: sample with replacement within each category
    - train SVC with linear kernel: c = 0.1

- statistical inference for p-vlue of each pixel

Computed on Zhou Workshop

@ Anmin Yang(UChicago), 2023-06-15
"""
import numpy as np
import pandas as pd
import os

import scipy.io
from scipy.stats import norm
import time

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV

def sample_data(X, y):
    """
    Sample with replacement within each category

    Parameters
    ----------
    X : numpy array, shape (n_samples, n_features)
        data matrix
    y : numpy array, shape (n_samples, 1)
        labels
    
    Returns
    -------
    X_sample : numpy array
        sampled data matrix
    y_sample : numpy array
    """
    X_sample = np.empty((0, X.shape[1]))
    for i in np.unique(y):
        X_sample = np.vstack((X_sample, 
                              X[np.random.choice(np.where(y==i)[0], 
                                                 size=np.sum(y==i), 
                                                 replace=True), :]))
    y_sample = np.repeat(np.unique(y), np.bincount(y.flatten())[1:])
    
    return X_sample, y_sample.reshape(-1, 1)

def CI(data_matrix,percentile=0.95):
    """
    calcuate the confidence interal of the bootstraped data
    
    Parameters
    ----------
    data_matrix: ndarray
        matrix in shape (m,n),
        m: number of bootstrap time 
        n: number of features 
    percentile: float
        the percntile of confidence interval, set to two sides
        default to 0.95 
    
    Returns
    -------
        ci: ndarray
            (m,3), m is the feature, equaling the n in matrix shape
                   3 is the CI and mean, (lower bound, upper bound,mean)
    """
    mean = np.mean(data_matrix,axis=0)
    std = np.std(data_matrix,axis=0)
    l_ci = norm.ppf((1-percentile)/2,mean,std)
    h_ci = norm.ppf((percentile+(1-percentile)/2),mean,std)
    
    ci = np.c_[l_ci.T,h_ci.T,mean]
    
    return ci     

def stat_infer(data_matrix, alpha=0.05, is_two_tail=True):
    """
    statistical inference whether a specific voxel is statistically significant,
    given alpha level
    the inference is based on the bootstrapped data 
    
    Parameters
    ----------
    data_matrix: ndarray
        shape (m,n)
        m: bootstrap time
        n: number of features
    alpha: float
        alpha level, 
        default at 0.05
    is_two_tail: bool
        whether the statistical inference is two-tail or one-tail
        default is two-tail 
    
    Returns
    -------
    bool_map: ndarray
        boolean value in each element
        shape (1,n)
    """
    if is_two_tail:
        percentile = 1 - alpha/2
    else:
        percentile = 1 - alpha
    
    ci = CI(data_matrix,percentile) # low_ci, high_ci, mean 
    
    bool_map = []
    for i in range(data_matrix.shape[1]):
        if (0>ci[i,0]) and (0<ci[i,1]):
            bool_map.append(False)
        else:
            bool_map.append(True)
    
    bool_map = np.array(bool_map).reshape(1,-1)
    return bool_map

def bootstrap_loop():
    boot_holder = {}
    for i in range(5):
        boot_holder[i] = np.empty((n_bootstrap, X_train.shape[1]))

    for i in range(n_bootstrap):
        start = time.time()
        X, y = sample_data(X_train, y_train)
        clf = OneVsRestClassifier(SVC(kernel='linear', 
                                    probability=True,
                                    C=0.1), 
                                n_jobs=5)
        clf.fit(X, y)
        counter_e = 0
        for estimator in clf.estimators_:
            weight = estimator.coef_.copy()
            # convert nan to 0
            weight[np.isnan(weight)] = 0
            boot_holder[counter_e][i,:] = weight 
            counter_e += 1
        
        end = time.time()
        print(f'bootstrap {i} finished, time elapsed: {end-start}s')

    # save the bootstrapped data
    np.save(os.path.join(data_path,
                            'results',
                            'boot_holder.npy'), boot_holder)
    return boot_holder

def stat_infer_loop(boot_holder):
    # statistical inference
    for key, value in boot_holder.items():
        p_map = stat_infer(value, alpha=0.05, is_two_tail=True)
        np.save(os.path.join(data_path, 
                            'results',
                            f'p_map_{key+1}.npy'), p_map) # name starts from 1

if __name__ == '__main__':
    data_path =  '/data/home/attsig/workingdir/svm_optic/optic_data'

    # load activation maps
    data = np.array(scipy.io.loadmat(os.path.join(data_path,
                                                'image_matrix_150.mat'))['image_matrix'])
    data = data.transpose() # the data has already been normalized 
    # load labels
    labels = np.array(scipy.io.loadmat(os.path.join(data_path,
                                                    'label.mat'))['label'])

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(data, labels, 
                                                        test_size=0.2, 
                                                        random_state=42,
                                                        stratify=labels)
    
    n_bootstrap = 1000
    boot_holder = bootstrap_loop()
    stat_infer_loop(boot_holder)