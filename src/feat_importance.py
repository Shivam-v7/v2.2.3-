# AMRUTA OXAI V2
# Copyright Amruta Inc. 2021
# Author: Dishit Kotecha

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
#import catboost as cb

def plot_importance(model, model_type, x):
    columns = x.columns
    if (model_type == 'Random Forest Classifier') or (model_type == 'Random Forest Regressor'):
        num_columns = len(columns)+1
        num_features = st.slider('Number of features to show.',
                                 min_value=1,
                                 max_value=int(max(range(num_columns))))
        #importances = model.feature_importances_
        #indices = np.argsort(importances)[::-1][:num_features]
        #features = columns

        fi = pd.DataFrame({'Variable':columns,
                   'Importance':model.feature_importances_}).sort_values('Importance', ascending=False)

        plt.title('Feature Importance')
        sns.barplot(x='Importance', y='Variable', data=fi[:num_features])
        #plt.barh(range(len(indices)), importances[indices], align='center')
        #plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        st.pyplot(bbox_inches='tight')
    elif (model_type == 'Linear Regression') or (model_type == 'Logistic Regression'):
        st.write('Intercept: ', model.intercept_)
        if model_type == 'Logistic Regression':
            coefs = model.coef_[0]
        else:
            coefs = model.coef_
        num_features = st.slider('Number of features to show.', min_value=1, max_value=int(max(range(len(columns)+1))))
        fi = pd.DataFrame({'Variable':columns,
                   'Importance':coefs}).sort_values('Importance', ascending=False)

        sns.barplot(x='Importance', y='Variable', data=fi[:num_features])
        plt.xticks(rotation=90)
        plt.xlabel('Estimated Coef.')
        st.pyplot(bbox_inches='tight')
    elif (model_type == 'XGBoost Classifier') or (model_type == 'XGBoost Regressor'):
    #elif any(x in model_type for x in ['XGBoost', 'LGBM']):
        st.write('Using the standard XGBOOST importance plot feature, exposes the fact that the most important feature is not stable, select'
             ' different importance types using the selectbox below')
        num_columns = len(columns)
        num_feat_input = st.slider('Number of features to show.',
                                   min_value=1,
                                   max_value=num_columns)
        importance_type = st.selectbox('Select the desired importance type', ('weight','gain','cover'),index=0)
        importance_plot = xgb.plot_importance(model,importance_type=importance_type, max_num_features=num_feat_input)
        plt.title ('XGBoost importance type = '+ str(importance_type))
        st.pyplot(bbox_inches='tight')
        plt.clf()
    elif ('Decision Tree' in model_type):
        num_columns = len(columns)
        num_features = st.slider('Number of features to show.',
                                 min_value=1,
                                 max_value=num_columns)
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:num_features]
        features = columns

        plt.title('Feature Importance')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        st.pyplot(bbox_inches='tight')
    elif (model_type == 'LGBM Classifier') or (model_type == 'LGBM Regressor'):
        num_columns = len(columns)
        num_feat_input = st.slider('Number of features to show.',
                                   min_value=1,
                                   max_value=num_columns,
                                   value=int((num_columns-1)/2))
        import_type = st.selectbox('Select desired importance type', ('split', 'gain'), index=0)

        plot = lgb.plot_importance(model, importance_type=import_type, max_num_features=num_feat_input)
        plt.title('LightGBM importance type='+str(import_type))
        st.pyplot(bbox_inches='tight')
    else:
        st.info('Not available for '+model_type+'. Try another explainability method.')
