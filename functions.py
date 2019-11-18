""" Some suppporting functions that are used by others notebooks """

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def print_confusion_matrix(y_train, y_pred):
    """ Display the confusion matrix for y_pred when compared with y_train  """
    
    cm = confusion_matrix(y_train, y_pred)
    df_cm = pd.DataFrame(cm, range(3), range(3))
    df_cm = pd.DataFrame(cm, columns=np.unique(y_train), index = np.unique(y_train))
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


def submit_to_competition(y_pred):
    """ Save the predicted values to submission.csv file 
        for final submission to the competition
    """

    submission = pd.read_csv('./dataset/submission_format.csv')
    submission['damage_grade'] = y_pred
    submission['damage_grade'] = submission['damage_grade'].astype(int)
    submission.to_csv('submission.csv', index=False)
    
    