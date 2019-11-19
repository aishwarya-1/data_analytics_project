""" Some suppporting functions that are used by others notebooks """

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV


def print_confusion_matrix(y_test, y_pred):
    """ Display the confusion matrix for y_pred when compared with y_test  """
    
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, range(3), range(3))
    df_cm = pd.DataFrame(cm, columns=np.unique(y_test), index = np.unique(y_test))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (10,7))
    sns.set(font_scale=1.4)    # for label size
    ax = sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})      # font size
    ax.get_ylim()
    ax.set_ylim(3.0, 0)


def knn_classifier(X_train,y_train,X_test,y_test = None,n_neighbors = 8):
    """ kNN Classifier used to train on the train set and predict for the test set
        Return the KNeighborsClassifier Model and y_pred (predicted values for X_test data)
        If y_test = None, model is used for predicting for final submission
    """

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if  y_test is not None:
        print("Accuracy of KNeighborsClassifier:", accuracy_score(y_test,y_pred))
    
    return (model, y_pred)


def random_forest_classifier(X_train,y_train,X_test,y_test = None,n_estimators = 200):
    """ Random Forest Classifier used to train on the train set and predict for the test set
        Return the RandomForestClassifier Model and y_pred (predicted values for X_test data)
        If y_test = None, model is used for predicting for final submission
    """

    model=RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if y_test is not None:
        print("Accuracy of RandomForestClassifier:", accuracy_score(y_test,y_pred))
    
    return (model, y_pred)


def xgboost_classifier(X_train,y_train,X_test,y_test = None,params=None):
    """ XGBoost Classifier used to train on the train set and predict for the test set
        Return the XGBClassifier Model and y_pred (predicted values for X_test data)
        If y_test = None, model is used for predicting for final submission
    """
    
    # defining default parameters for the XGBoost Model
    # the following hyper parameters were outputted from the xgboost_randomized_search() function
    if params is None:
        params = {
                    'learning_rate': 0.3,
                    'subsample': 0.8, 
                    'min_child_weight': 5, 
                    'max_depth': 10, 
                    'gamma': 0.3, 
                    'colsample_bytree': 0.8,
                    'objective': 'multi:softmax',
                    'num_class': 3,
                }
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if y_test is not None:
        print("Accuracy of XGBoostClassifier:", accuracy_score(y_test,y_pred))
        
    return (model,y_pred)


def voting_classifier(X_train,y_train,X_test,y_test = None,estimators=None,voting_type='soft'):
    """ Voting Classifier used for combining multiple classifiers models via voting techniques
        Return VotingClassifier Model and y_pred (produced after voting)
        If y_test = None, model is used for predicting for final submission
    """
    if estimators is None:
        raise ValueError('estimators(list) can`t be empty!')
    
    model = VotingClassifier(
        estimators = estimators, 
        voting = voting_type
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if y_test is not None:
        print("Accuracy of Ensemble of Classifiers:", accuracy_score(y_test,y_pred))
    
    return (model,y_pred)


def xgboost_randomized_search(X_train,y_train):
    """ Performs a randomized search on the hyper parameters of the XGBoost Model
        Returns an object containing the best parameters for the XGBoost Model
    """
    params={
            "learning_rate"    : [ 0.05, 0.10, 0.15, 0.20, 0.25, 0.30] ,
            "max_depth"        : [ 5, 6, 8, 9, 10, 12],
            "min_child_weight" : [ 0, 1, 2, 3.5, 5 ],
            "gamma"            : [ 0.0, 0.1, 0.25, 0.3, 0.4],
            "colsample_bytree" : [ 0.7, 0.75, 0.8, 0.85],
            "subsample"        : [ 0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
            "n_estimators"     : [ 100]
                
            }

    classifier = XGBClassifier(objective = 'multi:softmax')

    random_search = RandomizedSearchCV(classifier, param_distributions=params, n_iter=5, n_jobs=-1, cv=5, verbose=3)
    random_search.fit(X_train,y_train)
    best_params = random_search.best_params_
    return best_params


def submit_to_competition(y_pred):
    """ Save the predicted values to submission.csv file 
        for final submission to the competition
    """

    submission = pd.read_csv('./dataset/submission_format.csv')
    submission['damage_grade'] = y_pred
    submission['damage_grade'] = submission['damage_grade'].astype(int)
    submission.to_csv('submission.csv', index=False)
