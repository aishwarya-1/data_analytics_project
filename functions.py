""" Some suppporting functions used by others notebooks """

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score



def knn_classifier(X_train,y_train,X_test,y_test = None,n_neighbors = 8):
    """ kNN Classifier used to train on the train set and predict for the test set
        Return the KNeighborsClassifier Model and y_pred (predicted values for X_test data)
        If y_test = None, model is used for predicting for final submission
    """

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if  y_test is not None:
        print("Accuracy of KNeighborsClassifier: ", accuracy_score(y_test,y_pred))
    
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
        print("Accuracy of RandomForestClassifier: ", accuracy_score(y_test,y_pred))
    
    return (model, y_pred)


def submit_to_competition(y_pred):
    """ Save the predicted values to submission.csv file 
        for final submission to the competition
    """

    submission = pd.read_csv('./dataset/submission_format.csv')
    submission['damage_grade'] = y_pred
    submission['damage_grade'] = submission['damage_grade'].astype(int)
    submission.to_csv('submission.csv', index=False)