# AMRUTA OXAI V2
# Copyright Amruta Inc. 2021
# Author: Dishit Kotecha

########################################################
#################### MODELING ##########################
########################################################

from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from xgboost import XGBRegressor, XGBClassifier, DMatrix
import xgboost as xgb
from lightgbm.basic import LightGBMError
from lightgbm import LGBMClassifier, LGBMRegressor
# from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
import base64
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
import pandas as pd
import os, pickle
import plotly.express as px
import plotly.graph_objects as go
import matplotlib as mpl
import re
import plotly.figure_factory as ff

class Model():
    bi_classification_models = {'Logistic Regression': LogisticRegression(),
                            #'Support Vector Machine Classifier':SVC(gamma='auto',probability=True),
                            'Random Forest Classifier':RandomForestClassifier(n_estimators=10),
                            'Decision Tree Classifier':DecisionTreeClassifier(),
                            'XGBoost Classifier': XGBClassifier(),
                            'LGBM Classifier':LGBMClassifier()
                            # 'Cat Boost Classifier':CatBoostClassifier()
                            #'Stochastic Gradient Descent Classifier':SGDClassifier(loss='modified_huber'),
                            }
    multi_classification_models = {'Logistic Regression': LogisticRegression(multi_class='multinomial'),
                            #'Support Vector Machine Classifier':SVC(gamma='auto',probability=True),
                            'Random Forest Classifier':RandomForestClassifier(n_estimators=10),
                            'Decision Tree Classifier':DecisionTreeClassifier(),
                            'XGBoost Classifier': XGBClassifier(),
                            'LGBM Classifier':LGBMClassifier()
                            #'Cat Boost Classifier':CatBoostClassifier()
                            #'Stochastic Gradient Descent Classifier':SGDClassifier(loss='modified_huber'),
                            }

    regression_models = {'Linear Regression': LinearRegression(),
                        'XGBoost Regressor': XGBRegressor(objective='reg:squarederror'),
                        #'Support Vector Machine Regressor': SVR(probability=True),
                        'Random Forest Regressor': RandomForestRegressor(n_estimators=10),
                        'Decision Tree Regressor':DecisionTreeRegressor(),
                        'LGBM Regressor':LGBMRegressor()
                        #'Cat Boost Regressor':CatBoostRegressor()
                        #'Stochastic Gradient Descent Regressor':SGDRegressor()
                        }

    heuristic_models = ['Logistic Regression', 'Linear Regression']
    tree_based_models = ['XGBoost Classifier', 'XGBoost Regressor', 'Random Forest Classifier',
                         'Random Forest Regressor', 'Decision Tree Classifier', 'Decision Tree Regressor',
                         'LGBM Classifier', 'LGBM Regressor']

@st.cache(allow_output_mutation=True, hash_funcs=dict(dict=lambda dict: id(dict)))
def load_model(model):
    loaded_model = pickle.load(model)
    return loaded_model

#@st.cache(allow_output_mutation=True)
def prep_for_model(df, pred_var, train_split):
    """preps data for modeling depending on what model is chosen by user.
    splits data into training and test set.
    """
    x = df.drop(pred_var, axis=1)
    y = df[pred_var]

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 1-train_split, random_state = 42)

    return xtrain, xtest, ytrain, ytest

@st.cache(suppress_st_warning=True, allow_output_mutation=True, hash_funcs={'xgboost.sklearn.XGBRegressor': id, dict:id})
def train_and_predict_regression(xtrain, ytrain, xtest, model_type):
    """trains model with training data and predicts"""
    try:
        model = Model.regression_models[model_type]

        model.fit(xtrain, ytrain)
        ypred = model.predict(xtest)

        return ypred, model
    except ValueError as er:
        st.error(er)
    except LightGBMError as er:
        xtrain = xtrain.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        xtest = xtest.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        return train_and_predict_regression(xtrain, ytrain, xtest, model_type)


@st.cache(suppress_st_warning=True, allow_output_mutation=True, hash_funcs={XGBClassifier: id})
def train_and_predict_bi_classification(xtrain, ytrain, xtest, model_type):
    """trains model with training data and predicts"""
    try:
        model = Model.bi_classification_models[model_type]

        model.fit(xtrain, ytrain)
        ypred = model.predict(xtest)
        yscore = model.predict_proba(xtest)
        return ypred, yscore, model
    except ValueError as e:
        st.error(e)
    except LightGBMError as er:
        xtrain = xtrain.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        xtest = xtest.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        return train_and_predict_bi_classification(xtrain, ytrain, xtest, model_type)

@st.cache(suppress_st_warning=True, allow_output_mutation=True, hash_funcs={XGBClassifier: id})
def train_and_predict_multi_classification(xtrain, ytrain, xtest, model_type):
    """trains model with training data and predicts"""
    try:
        model = Model.multi_classification_models[model_type]

        model.fit(xtrain, ytrain)
        ypred = model.predict(xtest)

        yscore = model.predict_proba(xtest)
        return ypred, yscore, model
    except ValueError as e:
        st.error(e)
    except LightGBMError as er:
        xtrain = xtrain.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        xtest = xtest.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        return train_and_predict_multi_classification(xtrain, ytrain, xtest, model_type)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#@st.cache
def evaluate_regression_model(ytest, ypred):
    evs = metrics.explained_variance_score(ytest, ypred)
    maxe = metrics.max_error(ytest, ypred)
    mae = metrics.mean_absolute_error(ytest, ypred)
    mape = mean_absolute_percentage_error(ytest, ypred)
    mse = metrics.mean_squared_error(ytest, ypred)
    rmse = round(np.sqrt(metrics.mean_squared_error(ytest, ypred)), 2)
    r2 = metrics.r2_score(ytest, ypred)

    evaluation_df = pd.DataFrame({'Metric': ['Explained Variance Score','Max Error','Mean Absolute Error',
                                             'Mean Absolute Percentage Error','Mean Squared Error',
                                             'Root Mean Squared Error', 'R2'],
                                'Score':[evs, maxe,mae, mape, mse, rmse, r2]}).set_index('Metric')
    return evaluation_df

def plot_regression_results(ytest, ypred, pred_var):
    fig = go.Figure()
    to_plot = pd.DataFrame({'Actual': ytest,
                            'Predictions': ypred,
                            'Residuals': ytest - ypred})
    fig.add_trace(go.Scatter(x=to_plot.index, y=to_plot['Actual'],
                             mode='markers',
                             name='Actual'))
    fig.add_trace(go.Scatter(x=to_plot.index, y=to_plot['Predictions'],
                             mode='markers',
                             name='Predictions'))
    fig.update_layout(title='Actual vs. Predictions',
                      xaxis_title='Record',
                      yaxis_title=pred_var,
                      title_x=0.46,
                      autosize=False,
                      width=800,
                      height=500)

    st.plotly_chart(fig)
    # to_plot = pd.DataFrame({'Actual':ytest,
    #                         'Predictions':ypred,
    #                         'Residuals': ytest-ypred})
    #
    # to_plot[['Actual', 'Predictions']].plot(linestyle='', marker='o')
    # plt.title('Actual vs. Predictions')
    # plt.legend(loc='upper left')
    # plt.ylabel(pred_var)
    # plt.xlabel('Record')
    # st.pyplot(bbox_inches='tight')


    #residual
    plt.title('Residual Plot - Predictions vs Residuals')
    sns.residplot(to_plot['Predictions'], to_plot['Residuals'],
                  lowess=True, color='g')
    plt.ylabel('Standardized Residuals')
    plt.xlabel('Predictions')
    st.pyplot(bbox_inches='tight')


@st.cache(suppress_st_warning=True)
def evaluate_classification_model(ytest, ypred, yscore, model_type, model_method):
    """Views variety of model evaluation methods, including accuracy, recall, precision, confusion matrix and RMSE"""
    acc = metrics.accuracy_score(ytest, ypred)
    prec = metrics.precision_score(ytest, ypred, average='binary')
    rec = metrics.recall_score(ytest, ypred, average='binary')
    f1 = metrics.f1_score(ytest, ypred, average='binary')
    conf_matrix = metrics.confusion_matrix(ytest, ypred)
    print('evaluation')
    evaluation_df = pd.DataFrame({'Metric': [ 'Accuracy', 'Precision', 'Recall', 'F1'],
                                 'Score': [acc, prec, rec, f1]})


    # #ROC and AUC
    # falsepos, truepos, thresholds = metrics.roc_curve(ytest, yscore[:,1])
    # roc_auc = metrics.auc(falsepos, truepos)

    return evaluation_df, conf_matrix

@st.cache(suppress_st_warning=True)
def evaluate_multi_classification_model(ytest, ypred, yscore, model_type):
    """Views variety of model evaluation methods, including accuracy, recall, precision, confusion matrix and RMSE"""
    acc = metrics.accuracy_score(ytest, ypred)
    prec = metrics.precision_score(ytest, ypred, average='weighted')
    rec = metrics.recall_score(ytest, ypred, average='weighted')
    f1 = metrics.f1_score(ytest, ypred, average='weighted')
    conf_matrix = metrics.confusion_matrix(ytest, ypred)

    evaluation_df = pd.DataFrame({'Metric': [ 'Accuracy', 'Precision', 'Recall', 'F1'],
                                 'Score': [acc, prec, rec, f1]})

    return evaluation_df, conf_matrix

@st.cache
def get_roc_auc(ytest, yscore):
        #ROC and AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    ytest_dummies = pd.get_dummies(ytest, drop_first = False).values
    for i in list(set(ytest)):
        fpr[i], tpr[i], thresholds = metrics.roc_curve(ytest_dummies[:,i], yscore[:,i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    return fpr, tpr, roc_auc

def plot_conf_matrix(conf_matrix):
    fig = ff.create_annotated_heatmap(conf_matrix, x=list(range(len(conf_matrix))), y=list(range(len(conf_matrix))),
                                      colorscale='agsunset')

    # add title
    fig.update_layout(
        xaxis_title='Predicted',
        yaxis_title='Actual'
    )

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    fig['data'][0]['showscale'] = True
    st.plotly_chart(fig, bbox_inches='tight')
    plt.clf()

def plot_auc_roc(model_type, falsepos, truepos, roc_auc):
    st.subheader('ROC/AUC')
    plt.title(model_type)
    plt.plot(falsepos, truepos, 'b',
    label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    st.pyplot(bbox_inches='tight')

def plot_multi_auc_roc(model_type, fpr, tpr, roc_auc):
    fig = go.Figure()
    st.subheader('ROC/AUC')
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                             mode='lines',
                             name="No Skill",
                             line=dict(color='blue', width=2, dash='dash')))
    for i in list(fpr.keys()):
        fig.add_trace(go.Scatter(x=fpr[i], y=tpr[i],
                                 mode='lines+markers',
                                 name='ROC %0.2f for label %i' % (roc_auc[i], i)))
    fig.update_layout(title=model_type,
                      xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate',
                      title_x=0.43,
                      autosize=False,
                      width=800,
                      height=500)
    # fig, ax = plt.subplots()
    # ax.plot([0,1],[0,1],'r--')
    # for i in list(fpr.keys()):
    #     ax.plot(fpr[i], tpr[i], label='ROC %0.2f for label %i'%(roc_auc[i], i))
    # ax.set_title(model_type)
    # ax.legend(loc='lower right')
    # ax.set_xlim([-0.1,1.2])
    # ax.set_ylim([-0.1,1.2])
    # ax.set_ylabel('True Positive Rate')
    # ax.set_xlabel('False Positive Rate')
    # st.pyplot(bbox_inches='tight')
    st.plotly_chart(fig)

def plot_prec_recall(ytest, ytest_prob):
    prec, recall, thresholds = metrics.precision_recall_curve(ytest, ytest_prob)

    #get fscores
    fscore = (2 * prec * recall) / (prec + recall)

    # get best fscore and threshold value
    ix = np.argmax(fscore)
    best_threshold, best_fscore = thresholds[ix], fscore[ix]
    no_skill = len(ytest[ytest==1]) / len(ytest)

    # plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[no_skill, no_skill],
                             mode='lines+markers',
                             name='No Skill', line=dict(color='blue', width=2,
                                                        dash='dash')))
    fig.add_trace(go.Scatter(x=recall, y=prec,
                             mode='lines+markers',
                             name='Logistic'))
    fig.update_layout(title='Precision Recall Curve',
                      xaxis_title='Recall',
                      yaxis_title='Precision',
                      title_x=0.46,
                      autosize=False,
                      width=800,
                      height=500)
    # fig, ax = plt.subplots()
    # ax.set_title('Precision-Recall Curve')
    # ax.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    # ax.plot(recall, prec, marker='.', label='Logistic')

    # # axis labels
    # ax.set_xlabel('Recall')
    # ax.set_ylabel('Precision')
    #
    # # show the legend
    # ax.legend()

    # show the plot
    st.plotly_chart(fig)
    return best_threshold, best_fscore


def download_model(model):
    output_model = pickle.dumps(model)
    b64 = base64.b64encode(output_model).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" class="button" download="my_model.pkl">Download Trained Model .pkl File</a>'
    st.markdown(href, unsafe_allow_html=True)

def download_predictions(xtest, predictions, actual):
    output_df = xtest.copy()
    output_df['Prediction'] = predictions
    output_df['Actual'] = actual
    csv = output_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" class="button" download="predictions.csv">Download Test Set Predictions CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)

def download_processed_data(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" class="button" download="processed_data.csv">Download Processed CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)

##### CROSS VALIDATION #####

# kfold
def kfold_cv(x, y):
    kf = KFold(n_splits=2, n_folds=4, random_state=42)
    kf.get_n_splits(x)
    for train_index, test_index in kf.split(x):
        print('TRAIN:', train_index, 'TEST:', test_index)

def get_cv_score(model, x, y, model_method, splits):
    if 'Classification' in model_method:
        return cross_val_score(model, x, y, scoring='accuracy', cv=splits)
    elif 'Regression' in model_method:
        return cross_val_score(model, x, y, scoring='r2', cv=splits)

def get_cv_predictions():
    return

def plot_cv(cv_scores, model_method):
    fig = go.Figure()
    # fig, ax = plt.subplots()
    scores = dict(enumerate(cv_scores, 0))
    if 'Classification' in model_method:
        fig.add_trace(go.Scatter(x=list(scores.keys()), y=list(scores.values()), mode='lines+markers'))
        fig.update_layout(xaxis_title='Split',
                        yaxis_title='Accuracy',
                        autosize=False,
                        width=800,
                        height=500,
                        xaxis=dict(tickmode="linear"))
        # ax.plot(cv_scores, marker='o', linestyle='-')
        # ax.set_xlabel('Split')
        # ax.set_ylabel('Accuracy')
        # return fig
    elif 'Regression' in model_method:
        fig.add_trace(go.Scatter(x=list(scores.keys()), y=list(scores.values()), mode='lines+markers'))
        fig.update_layout(xaxis_title='Split',
                          yaxis_title='R2',
                          autosize=False,
                          width=800,
                          height=500,
                          xaxis=dict(tickmode="linear"))
        # ax.plot(cv_scores, marker='o', linestyle='-')
        # ax.set_xlabel('Split')
        # ax.set_ylabel('R2')
        return fig
    st.plotly_chart(fig)
