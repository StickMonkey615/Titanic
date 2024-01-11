'''Use Machine Learning algorithms to predict which passengers on the Titanic sinking based on individual passenger
information. Based on the Kaggle challenge at https://www.kaggle.com/competitions/titanic.'''

# Import dependencies
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import imblearn

# Import datasets
train_df = pd.read_csv("train.csv") 
test_df = pd.read_csv("test.csv")

from processing import process_df
processed_train_df = process_df(train_df)
processed_test_df = process_df(test_df, test=True)

# Split your data so that you can test the effectiveness of your model
# Split the data into a Training set and a Test set
dfs = np.split(processed_train_df, [len(processed_train_df.columns)-1], axis=1)
X = dfs[0]
y = dfs[1]
X_cols = processed_train_df.columns[0:-1]
y_cols = processed_train_df.columns[-1]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.2,
                                                    random_state = 42)
y_train=y_train.astype('bool')
y_test=y_test.astype('bool')
del dfs, X, y

# Apply feature scaling
from processing import apply_scaling
X_train_proc = apply_scaling(X_train)
X_test_proc = apply_scaling(X_test)
del X_train, X_test

# Apply dimensionality reduction
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 8, kernel = 'rbf')
X_train_proc = kpca.fit_transform(X_train_proc)
X_test_proc = kpca.transform(X_test_proc)

# Create models for evaluation
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.ensemble import EasyEnsembleClassifier, BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import accuracy_score,log_loss,recall_score,balanced_accuracy_score
from sklearn.metrics import precision_score,f1_score,confusion_matrix, fbeta_score
from imblearn.metrics import sensitivity_score, specificity_score, geometric_mean_score

models = ['ADA',
          'GBC'
        #   'EEC',
        #   'BRF',
        #   'RFC',
        #   'KNN',
        #   'SVC',
        #   'lReg'
          ]

classifiers = [AdaBoostClassifier(random_state=42),
               GradientBoostingClassifier(random_state=42)
            #    EasyEnsembleClassifier(estimator=RandomForestClassifier(random_state=42,
            #                                                                 n_estimators=60)),
            #    BalancedRandomForestClassifier(random_state=42),
            #    RandomForestClassifier(random_state=42, n_jobs=-1),
            #    KNeighborsClassifier(n_jobs=-1),
            #    SVC(random_state=42, probability=True),
            #    LogisticRegression(solver='newton-cg', multi_class='multinomial')
               ]

# n_estimators range limited for EEC and KNN to reduce processing time. Initially
# ranges from 10 to 100 (incl.) were run before smaller ranges were selected.
params = {models[0]:{'learning_rate':[0.01,0.1,1,10],
                     'n_estimators':np.array(range(10,110,10)),
                     'algorithm':['SAMME','SAMME.R']},
          models[1]:{'learning_rate':[0.01,0.1,1,10],
                     'n_estimators':np.array(range(10,110,10)),
                     'max_depth':np.array(range(1,11,1))},
        #   models[2]:{'n_estimators':np.array(range(10,60,10)),
        #              'estimator__n_estimators':np.array(range(40,100,10)),
        #              'estimator__criterion':['gini','entropy'],
        #              'estimator__class_weight':['balanced','balanced_subsample']},
        #   models[3]:{'n_estimators':np.array(range(10,110,10)),
        #              'criterion':['gini','entropy'],
        #              'class_weight':['balanced','balanced_subsample']},
        #   models[4]:{'n_estimators':np.array(range(10,110,10)),
        #              'criterion':['gini','entropy'],
        #              'class_weight':['balanced','balanced_subsample']},
        #   models[5]:{'n_neighbors':np.array(range(10,60,10)),
        #              'weights':['uniform','distance'],
        #              'algorithm':['auto','ball_tree'],
        #              'metric':['chebyshev','minkowski']},
        #   models[6]:{'C':[1,10,60,100,600,1000],
        #              'tol':[0.005],
        #              'kernel':['linear','poly','rbf','sigmoid'],
        #              'class_weight':['balanced']},
        #   models[7]:{'C':[1,10,60,100,600,1000],
        #              'class_weight':['balanced'],
        #              'solver':['newton-cg','lbfgs','sag','saga'],
        #              'tol':[0.0001]}
                     }

# Define custom scoring metric to be used to decide between different
# classifiers after grid search.
from sklearn.metrics import make_scorer
import math
def my_scorer(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    score = tn/(tn+fn) #measure between false negative and true negatives
    if math.isnan(score):
        score = 0.01
    return score

cust_score = make_scorer(my_scorer,greater_is_better=True)

# Create interim csv to diagnose issue with clf.fit returning 'unknown' type error
interim = pd.DataFrame(X_train_proc)
interim.to_csv(f"Interim_x_train.csv")

# Conduct halving grid search across all models
y_tested=0
test_scores=[]
for name, estimator in zip(models,classifiers):
    print(name)
    clf = HalvingGridSearchCV(estimator=estimator,
                            param_grid=params[name],
                            factor=2,
                            scoring='balanced_accuracy',
                            cv=5,
                            n_jobs=-1,
                            verbose=0)
    clf.fit(X_train_proc, np.ravel(y_train.values))
    estimates = clf.predict_proba(X_test_proc)
    y_tested+=estimates
    acc = accuracy_score(y_test, clf.predict(X_test_proc))
    rec = recall_score(y_test, clf.predict(X_test_proc))
    pre = precision_score(y_test, clf.predict(X_test_proc))
    f1s = f1_score(y_test, clf.predict(X_test_proc), average='macro')
    cm = confusion_matrix(y_test, clf.predict(X_test_proc))
    sel_score = my_scorer(y_test, clf.predict(X_test_proc))    
    test_scores.append((name,acc,clf.best_score_,f1s,rec,pre,cm,clf.best_params_,sel_score))
    
submission = pd.DataFrame(test_scores, columns=['Classifier',
                                                'Accuracy',
                                                'Trg balanced accuracy score',
                                                'F-score test',
                                                'Recall',
                                                'Precision',
                                                'Confusion matrix',
                                                'Best params',
                                                'Selector'])
submission.to_csv(f"Results.csv")
del acc,clf,estimates,y_tested,rec,pre,f1s,cm,sel_score,test_scores,name,estimator