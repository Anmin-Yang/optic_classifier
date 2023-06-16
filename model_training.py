"""
Apply SVM with linear kernel to activation maps
1. Load activation maps
2. Apply SVM with linear kernel in one-vs-rest fashion
3. Save results
    - SVM weights
    - SVM matrices
        - accuracy

Computed locally

@ Anmin Yang(UChicago), 2023-06-14
"""
import numpy as np
import pandas as pd 
import os

import scipy.io
import time

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV

# load data 
data_path = '/Users/anmin/Documents/optic_data'

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

# SVM with linear kernel in one-vs-rest fashion
# grid search with k-fold for hyperparameters -C
start = time.time()
clf = OneVsRestClassifier(SVC(kernel='linear', probability=True), n_jobs=5)
parameters = {
    "estimator__C": [0.1, 1, 10, 100, 1000]
}
clf_tuned = GridSearchCV(clf, param_grid=parameters, cv=5, n_jobs=5,
                         scoring='accuracy')
clf_tuned.fit(X_train, y_train)
print(clf_tuned.best_params_) # c = 0.1 , fixed for bootstrap 
print(clf_tuned.best_score_)
end = time.time()
print('Time elapsed: ', end-start)

# save coefficients 
weights = np.empty((clf_tuned.best_estimator_.n_classes_,
                    clf_tuned.best_estimator_.n_features_in_))

for estimator in clf_tuned.best_estimator_.estimators_:
    weights[estimator.classes_[0], :] = estimator.coef_
np.save(os.path.join(data_path, 
                     'results',
                     'weights_point.npy'), weights)




 





 
