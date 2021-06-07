"""
AMRUTA OXAI V2
Copyright Amruta Inc. 2021
Author: Dishit Kotecha

Features:
* Upload your own dataset
* Dataset info and summary
* Train/Test Split
* Modeling
    - Logistic Regression
    - Random Forest
    - Decision Tree
    - XGBoost
    - Light GBM
    - CatBoost
* Model Evaluation
    - Regression
        -
    - Classification
        - Accuracy, Precision, Recall, F1
        - Confusion Matrix
        - ROC/AUC Curve
* Explainability
    - SHAP
    - feature importance
    - tree visualizer
    - correlation graph
    - tree interpreter
    - ice plots
    - pdp plots

Features to add:
* fill in missing data
* encode categorical variables - DONE
* Work with Multiclass classfication - DONE
* Correlation graph - DONE
* ICE Plots = DONE
* SFIT (Single Feature Introduction Test)
* connect to dataset hosted on cloud: GCP, Azure, AWS
* user login/authentication - DONE
* add info based on user selected input to sidebar
* add light gbm

TODO:
* need to define hash function for XGBRegressor/XGBClassifier under SessionState - DONE
* Categorize Explainability methods - DONE
    - Global Summary
    - Local Explanation
    - PDP/Interaction Plots
    - Feature Importance
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
from src.helpers import *
from src.modeling import *
from src.shap_exp import *
from src.ice_pdp import *
from src.correlation_graph import *
from src.lime_eli5 import *
from src.feat_importance import *
#from sfit import *
from datetime import datetime, date
import streamlit.components.v1 as components

from src.SessionState import *
from src.account_management import validate

# set seaborn theme
sns.set()

# # st.set_option('deprecation.showfileUploaderEncoding', False)
# st.set_option('deprecation.showPyplotGlobalUse', False)

DEMO_REGRESSION_DATA = 'sample_data/BostonHousing.csv'
DEMO_CLASSIFICATION_DATA = 'sample_data/titanic.csv'
DEMO_MULTI_CLASS_DATA = 'sample_data/iris.csv'
LOGO_IMAGE = 'images/amrutalogo2.png'
UPLOADED_DATASET = None
UPLOADED_MODEL = None

# declare session state
state = get_state({XGBClassifier:id, XGBRegressor:id})


def render_app():
    st.sidebar.title('Amruta XAI V2')

    ####### PAGE SELECTION ########
    pages = {
        'Data Explorer': data_exploration,
        'Data Processor': feature_engineering,
        'ML Modeler': modeling,
        'ML Explainer': explainability,
        'Notes': notes
    }

    sample_data_select = st.sidebar.selectbox('Select Sample data:',
                                          ['Regression: Boston Housing',
                                           'Bi-Classification: Titanic Survivors',
                                           'Multi-Classification: Iris Flowers',
                                           'None'])

    dataset_shape = st.sidebar.empty()
    separator = st.sidebar.selectbox('Select separator used in your dataset', ['\t', ',', '|', '$', ' '], 1)
    UPLOADED_DATASET = st.sidebar.file_uploader('Upload your preprocessed dataset in CSV format (Size Limit: 1028mb)', type='csv')
    UPLOADED_MODEL = st.sidebar.file_uploader('And/Or upload your pre-trained model (.sav or .pkl) (Size Limit: 1028mb)', type=['sav', 'pkl'])

    tab = st.sidebar.radio('Select Tab', list(pages.keys()))
    st.sidebar.image(Image.open(LOGO_IMAGE), width = 300)
    st.sidebar.text('Copyright Amruta Inc. 2021')
    #st.sidebar.text('Beta/Test Version')
    #st.sidebar.text('The purpose of this version is to test \nnew features.')
    st.sidebar.text("Logged in as %s" % state.user_name)

    log_out = st.sidebar.button('Log out')
    if log_out:
        state.user_name = None

    ## dataset selection
    df = None
    if UPLOADED_DATASET is not None:
        UPLOADED_DATASET.seek(0)
        sample_data_select = 'None'
        data_load_state = st.text('Loading data...')
        data_to_load = UPLOADED_DATASET
        data_load_state.text('Loading data... done!')
    else:

        if sample_data_select == 'Regression: Boston Housing':
            data_to_load = DEMO_REGRESSION_DATA

        elif sample_data_select == 'Bi-Classification: Titanic Survivors':
            data_to_load = DEMO_CLASSIFICATION_DATA

        elif sample_data_select == 'Multi-Classification: Iris Flowers':
            data_to_load = DEMO_MULTI_CLASS_DATA
        else:
            st.info('Please select a sample dataset or upload a dataset.')
            st.stop()

    ### LOAD DATA
    if ((df is None) and (state.processed_df is None)) or ((state.current_fn != data_to_load) and (UPLOADED_DATASET is None)):
        if data_to_load:
            try:
                df = load_data(data_to_load, separator)
                state.processed_df = None
                state.current_fn = data_to_load
            except FileNotFoundError:
                st.error('File not found.')
            except:
                st.error('Make sure you uploaded the correct file format.')
        else:
            st.info('Please upload some data or choose sample data.')

    ### LOAD MODEL
    user_model = None
    if UPLOADED_MODEL is not None:
        try:
            UPLOADED_MODEL.seek(0)
            user_model = load_model(UPLOADED_MODEL)
        except Exception as e:
            st.error(e)

    ## view dataset rows and columns in sidebar
    if state.processed_df is None:
        dataset_shape.text('Dataset shape\n Rows: %s\n Columns:%s' % (str(df.shape[0]), str(df.shape[1])))
    else:
        dataset_shape.text('Dataset shape\n Rows: %s\n Columns:%s' % (str(state.processed_df.shape[0]), str(state.processed_df.shape[1])))

    ########## TAB SELECTION ############
    if tab == 'Data Explorer':
        if (state.processed_df is not None) and not (state.processed_df.equals(df)):
            data_exploration(state.processed_df)
        else:
            data_exploration(df)
    elif tab == 'Data Processor':
        if state.processed_df is not None:
            state.processed_df = feature_engineering(state.processed_df)
        else:
            state.processed_df = feature_engineering(df)
    elif tab == 'ML Modeler':

        if state.processed_df is not None:
            try:
                state.model, state.xtrain, state.xtest, state.ytrain, state.ytest, state.ypred = modeling(state.processed_df, user_model)
            except TypeError as e:
                print(e)
        else:
            try:
                state.model, state.xtrain, state.xtest, state.ytrain, state.ytest, state.ypred = modeling(df, user_model)
            except TypeError as e:
                pass
    elif tab == 'ML Explainer':
        if state.model is not None:
            if state.processed_df is not None:
                explainability(state.model, state.xtrain, state.xtest, state.ytrain, state.ytest, state.ypred, state.processed_df)
            else:
                explainability(state.model, state.xtrain, state.xtest, state.ytrain, state.ytest, state.ypred, df)
        else:
            st.error('Please fit a model through the "ML Modeler" tab')
    elif tab == 'Notes':
        notes()



def data_exploration(df):

    st.header('Data Explorer')

    view_data_button = st.button('View All Data')
    if view_data_button:
        st.write(df)
    else:
        st.subheader('First Five Rows')
        st.write(df.head())

        st.subheader('Summary on Continuous attributes')
        st.write(df_summary(df))

        # view info categorical variables
        st.subheader('Summary on Categorical attributes')
        num_cols = df._get_numeric_data().columns

        categorical_features = list(df.select_dtypes(include=['category', 'object']).columns)
        cat = st.selectbox('Pick categorical feature to preview', ['None'] + categorical_features)
        if cat != 'None':
            value_counts = df[cat].value_counts()
            fig_plotly = go.Figure([go.Bar(x=value_counts.index, y=value_counts.values)])
            fig_plotly.update_layout(title='Frequency Plot of ' + cat, xaxis_title=cat, yaxis_title='# of Occurences', title_x=0.5)
            # sns.barplot(value_counts.index, value_counts.values)
            # plt.title('Frequency Plot of ' + cat)
            # plt.xlabel(cat)
            # plt.ylabel('# of Occurences')
            # st.pyplot(bbox_inches='tight')
            st.plotly_chart(fig_plotly)

        st.subheader('Columns and Records')
        rows, cols = df.shape
        st.write('There are %d records and %d columns/attributes.'%(rows, cols))
        st.write(df.columns)

        # heat map/correlation matrix
        # check_hm = st.checkbox('Create heatmap')
        # if check_hm:
        #     st.subheader('Correlation Heatmap')
        #     heatmap_load_state = st.text('Creating heatmap...')
        #     render_heatmap(df)
        #     heatmap_load_state.text('Creating heatmap... done!')

        ## view null values



        ###################################################

        d_null_count = pd.DataFrame(df.isnull().sum().reset_index().rename(columns={'index': 'Columns'}))
        d_null_count.columns = [*d_null_count.columns[:-1], 'Null Count']
        d_null_count['Null Percentage'] = (df.isna().sum()/(len(df))*100).to_list()
        d_null_count.drop(d_null_count[d_null_count['Null Count'] == 0].index, inplace=True)
        # st.dataframe(d_null_count)
        if not d_null_count.empty:
            st.subheader('Null Value Analysis')
            fig_plotly = go.Figure(data=[go.Bar(x=d_null_count['Columns'].values, y=d_null_count['Null Count'].values, hovertemplate='Count: %{y}<br>' + 'Percentage: %{text} <extra></extra>', text=['{:.2f}%'.format(i) for i in d_null_count['Null Percentage'].to_list()], name="Null", marker_color='red'),
                                    go.Bar(x=d_null_count['Columns'].values, y=(len(df)-d_null_count['Null Count']).values, hovertemplate='Count: %{y}<br>' + 'Percentage: %{text} <extra></extra>', text=['{:.2f}%'.format(100 - i) for i in d_null_count['Null Percentage'].to_list()], name="Non - Null", marker_color='blue')])
            fig_plotly.update_layout(xaxis_title='Columns', yaxis_title='Null Count',
                                     title_x=0.5, yaxis=dict(range=[0, len(df)]), barmode='group')
            st.plotly_chart(fig_plotly)

def feature_engineering(df):
    st.markdown('''
    Click on a drop down option to implement pre-processing steps.
    ''')
    processed_df = df.copy()

    ## index/slicing
    #st.subheader('Index/Slice DataFrame')
    with st.beta_expander('Index/Slice DataFrame'):
        view_desc1_button = st.button('View Description', key=301)
        if view_desc1_button:
            '''
            Subset dataframe by its index. For example, for a dataset of size 100, choosing
            10 as the head index and 100 as the tail index will ignore the first 10 records. Choosing
            0 as the head index and 50 as the tail index will only subset the first 50 records.
            '''

        print(state.slice_values)
        state.slice_values = st.slider('Select index range',
                                    min_value=0,
                                    max_value=df.shape[0],
                                    value=state.slice_values if state.slice_values
                                    and (0 <= state.slice_values[0] <= df.shape[0])
                                    and (state.slice_values[0] <= state.slice_values[1] <= df.shape[0])
                                    else [0, df.shape[0]])

        slice_button = st.checkbox('Slice')
        if slice_button:
            try:
                df_sliced = df.iloc[state.slice_values[0]:state.slice_values[1]]
                st.success('DataFrame sliced!')
            except:
                st.error('Dataframe unable to slice.')
        else:
            df_sliced = df
    #--------------------------------------------------------------------------------------------------#

    ##drop columns
    # st.subheader('Drop any columns')
    with st.beta_expander('Drop Columns'):
        view_desc2_button = st.button('View Description', key=302)
        if view_desc2_button:
            '''
            Drop any columns/features that are unnecessary for fitting the model. This may include ID, names, and any
            other features that serve as identification or are redundant.
            '''

        state.drop_cols = st.multiselect('Select columns to drop.',
                                        list(df.columns),
                                        state.drop_cols
                                        if state.drop_cols in list(df.columns)
                                        else None)

        drop_button = st.checkbox('Drop')
        if drop_button:
            try:
                df_dropped = df_sliced.drop(state.drop_cols, axis=1)
                if len(state.drop_cols) > 0:
                    st.success('Column(s) %s dropped!'%(str(state.drop_cols)))
            except:
                st.error('Failed to drop columns in dataframe.')
        else:
            df_dropped = df_sliced
    #---------------------------------------------------------------------------------------------------#
    ## Null Value Imputation

    # choose which columns to impute

    # for each column/feature, choose imputation method
    # must choose whether it must impute continuous values or categorical

    with st.beta_expander('Null Value Imputation'):
        state.null_columns = df_dropped.columns[df_dropped.isnull().any()].to_list()
        state.imputer_method = [0]*len(state.null_columns)
        state.column_type_null = ['']*len(state.null_columns)
        state.custom_impute = [0]*len(state.null_columns)
        column_type_options = ['Please select one', 'Numeric', 'Categorical']
        for i in range(len(state.null_columns)):
            st.write(state.null_columns[i])
            b2_col1, b2_col2 = st.beta_columns((1, 2))
            state.column_type_null[i] = b2_col1.selectbox('Column type',
                                                  column_type_options,
                                                  column_type_options.index(state.column_type_null[i])
                                                  if state.column_type_null[i]
                                                  else 0, key=state.null_columns[i])
            if state.column_type_null[i] == 'Please select one':
                st.info('Please select a column type')
                # st.stop()
            if state.column_type_null[i] == 'Categorical':
                imputer_options = ['Please select imputer', 'None', str(df_dropped[state.null_columns[i]].mode()[0]), 'Enter custom value']
                state.imputer_method[i] = b2_col2.selectbox('Imputer', imputer_options, key=state.null_columns[i])
            elif state.column_type_null[i] == 'Numeric':
                imputer_options = ['Please select imputer', 'Iterative Imputer']
                state.imputer_method[i] = b2_col2.selectbox('Imputer', imputer_options, key=state.null_columns[i])
            if state.imputer_method[i] == 'Enter custom value':
                state.custom_impute[i] = st.text_input("Custom Value to impute", state.custom_impute[i] if state.custom_impute[i] else '', key=state.null_columns[i])
                state.imputer_method[i] = state.custom_impute[i]
        impute_button = st.checkbox('Impute')
        if impute_button:
            try:
                df_null_imputed = null_impute(df_dropped, state.null_columns, state.imputer_method)
                st.success('Columns with a imputer selected were imputed')
            except Exception as e:
                st.error('Failed to impute. Please check column types.')
                st.write(e)
                df_null_imputed = df_dropped
        else:
            df_null_imputed = df_dropped

    #---------------------------------------------------------------------------------------------------#

    ## filter by column
    # st.subheader('Filter by Column Value')
    with st.beta_expander('Filter by Column Value'):
        view_desc3_button = st.button('View Description', key=303)
        if view_desc3_button:
            '''
            Filter dataframe by a column's value. Select a column and verify whether it is a categorical
            or numerical column.
            '''

        column_options = list(df_null_imputed.columns)
        column_type_opts = ['Numeric', 'Categorical']

        state.column_filt = st.selectbox('Select Column to filter by',
                                        column_options,
                                        column_options.index(state.column_filt)
                                                if (state.column_filt)
                                                    and (state.column_filt in column_options)
                                                    else 0)
        state.column_type_filt = st.selectbox('Column type',
                                            column_type_opts,
                                            column_type_opts.index(state.column_type_filt)
                                                if state.column_type_filt
                                                else 0)

        if state.column_type_filt == 'Numeric': # filter by numeric
            numeric_ops = ['>', '<', '=']
            state.compare = st.selectbox(' ', numeric_ops,
                                        numeric_ops.index(state.compare)
                                            if (state.compare)
                                            and (state.compare in numeric_ops)
                                            else 0)

            min_value = float(min(df[state.column_filt]))
            max_value = float(max(df[state.column_filt]))
            state.compare_val = st.number_input('Filter numeric',
                                                min_value=min_value,
                                                max_value=max_value,
                                                value=state.compare_val_num if (state.compare_val_num)
                                                        and (min_value <= state.compare_val_num <= max_value)
                                                        else min_value,
                                                format='%f')

        elif state.column_type_filt == 'Categorical': # filter by categorical
            cat_ops = ['contains', 'does not contain']
            state.compare = st.selectbox(' ',
                                        cat_ops, cat_ops.index(state.compare)
                                        if (state.compare)
                                            and (state.compare in cat_ops)
                                            else 0)
            cat_set_values = list(set(df[state.column_filt]))
            state.compare_val = st.selectbox('Filter text',
                                            state.compare_val_cat
                                                if (state.compare_val_cat)
                                                and (state.compare_val in cat_set_values)
                                                else 0)

        filter_button = st.checkbox('Filter')
        if filter_button:
            try:
                df_filtered = filter_df(df_null_imputed, state.compare, state.column_filt, state.compare_val)
                st.success('DataFrame Filtered!')
            except:
                st.error('Dataframe unable to filter.')
        else:
            df_filtered = df_null_imputed
    #---------------------------------------------------------------------------------------------------#

    ## encode features
    # st.subheader('Label Encode columns')
    with st.beta_expander('Label Encoding'):
        view_desc4_button = st.button('View Description', key=304)
        if view_desc4_button:
            '''
            Certain categorical columns that contain text attributes will need to be encoded into integer values. XAI V2
            provides 2 methods of encoding: Label Encoding and One-Hot Encoding.

            **Label Encoding** will transform each text value into an integer value and output a table of what
            the integer encoding stands for.

            **One-Hot Encoding** will transform each text value into its own separate column and label
            as 0 or 1. If the record contains that text value, it will be marked as 1, otherwise 0. This is recommended
            for features with high number of categories. Keep in mind that One-Hot Encoding could drastically increase
            the dimensionality of your dataset, which would affect model training time.
            '''

        encode_meth_options = ['Label Encoder', 'One Hot Encoder']
        state.encode_method_select = st.selectbox('Select Encoding Method',
                                            encode_meth_options,
                                            encode_meth_options.index(state.encode_method_select) if state.encode_method_select else 0)
        state.cols_to_encode = st.multiselect('Select Columns to encode',
                                            list(df_filtered.columns), state.cols_to_encode if state.cols_to_encode in list(df.columns) else None)
        encode_button = st.checkbox('Encode')
        if encode_button:
            try:
                if state.encode_method_select == 'Label Encoder':
                    df_encoded = label_encode(df_filtered, state.cols_to_encode)
                elif state.encode_method_select == 'One Hot Encoder':
                    df_encoded = pd.get_dummies(df_filtered, columns = state.cols_to_encode)
                st.success('Column(s) %s encoded!'%(str(state.cols_to_encode)))
            except:
                st.error('Failed to encode columns.')
        else:
            df_encoded = df_filtered

    #---------------------------------------------------------------------------------------------------#

    ## standard scaling numeric values
        # st.subheader('Standard Scaler')
    with st.beta_expander('Continuous Feature Normalization'):
        view_desc5_button = st.button('View Description', key=305)
        if view_desc5_button:
            '''
            Standardize features by removing the mean and scaling to unit variance

            The standard score of a sample x is calculated as:

            z = (x - u) / s
            Standardization of a dataset is a common requirement for many machine learning estimators:
            they might behave badly if the individual features do not more or less look like standard
            normally distributed data (e.g. Gaussian with 0 mean and unit variance).
            '''

        state.cols_to_scale = st.multiselect('Select Columns to scale',
                                            list(df_encoded.columns),
                                            state.cols_to_scale if state.cols_to_scale in list(df_encoded.columns)
                                            else None)

        scale_button = st.checkbox('Scale')
        if scale_button:
            try:
                df_scaled = standard_scale(df_encoded, state.cols_to_scale)
                st.success('Column(s) %s Scaled!'%(str(state.cols_to_scale)))
            except:
                st.error('Failed to standard scale columns.')
        else:
            df_scaled = df_encoded

    st.subheader('Processed Data Preview')
    st.write(df_scaled.head(20))
    download_processed_data(df_scaled)

    process_confirm_button = st.button('Confirm Processing Steps')
    if process_confirm_button:
        processed_df = df_scaled
        st.success('Processing steps applied!')

    return processed_df


def modeling(df, user_model):

    st.header('Modeling')

    model_method_opts = ['None', 'Binary Classification', 'Multi Class Classification', 'Regression']
    state.model_method = st.selectbox('Choose method', model_method_opts, model_method_opts.index(state.model_method) if state.model_method else 0)

    # choose variable to predict
    state.pred_var = st.selectbox('Select Predictive Variable', list(df.columns), list(df.columns).index(state.pred_var) if state.pred_var in list(df.columns) else 0)
    state.xcolumns = df.drop(state.pred_var, axis=1).columns
    st.write('Selected:', state.pred_var)

    if state.model_method != 'None':
        ############### UPLOADED MODEL #######################
        if user_model is not None:
            st.subheader('Using pre-trained model')
            st.write(type(user_model))
            x = df.drop(state.pred_var, axis=1)
            y = df[state.pred_var]
            st.write('Pre-trained model score:', user_model.score(x,y))
            st.markdown(state.model)


            ypred = user_model.predict(x)

            #download predictions
            download_predictions(x, ypred, y)

            state.model_type = st.selectbox('What Algorithm is this?', ['None'] + list(Model.bi_classification_models.keys()) + list(Model.regression_models.keys()))
            if state.model_type in list(Model.bi_classification_models.keys()):
                try:
                    ypred = user_model.predict(x)
                    st.write("ypred",ypred)
                    yscore = user_model.predict_proba(x)
                    st.write("yscore",yscore)
                    eval_df, conf_matrix, falsepos, truepos, roc_auc = evaluate_classification_model(y, ypred, yscore, state.model_type, state.model_method)

                    st.header('Model Evaluation')
                    st.table(eval_df)
                    st.subheader('Confusion Matrix')
                    plot_conf_matrix(conf_matrix)

                    plot_auc_roc(state.model_type, falsepos, truepos, roc_auc)
                except:
                    st.error('Your model cannot output predicted probabilities.')
            elif state.model_type in list(Model.regression_models.keys()):
                try:
                    ypred = user_model.predict(x)
                    eval_df = evaluate_regression_model(y, ypred)
                    st.table(eval_df)
                except:
                    pass

            return user_model, x, x, y, y, ypred
        ################# FITTING A MODEL ####################
        else:
            # train test split
            state.train_split = st.number_input('How much of data do you want to use for training set? (0 if you want to predict entire dataset)',
                                        min_value = 0.1,
                                        max_value = 1.0, value=state.train_split if state.train_split else 0.7)

            x = df.drop(state.pred_var, axis=1)
            y = df[state.pred_var]
            xtrain, xtest, ytrain, ytest = prep_for_model(df, state.pred_var, state.train_split)

            # cross validation
            state.cv_button = st.checkbox('Cross Validation (4 splits)', state.cv_button)
            '''
            Cross validation allows you to check if for underfitting or overfitting the data.
            Read more on Cross validation here: https://towardsdatascience.com/why-and-how-to-cross-validate-a-model-d6424b45261f
            '''

            ## Modeling ##
            if state.model_method == 'Binary Classification':
                if len(set(df[state.pred_var])) != 2:
                    st.error('Variable does not contain binary values to predict. This will lead to errors.')
                st.subheader('Classification')

                ## Modeling
                class_model_opts = list(Model.bi_classification_models.keys())
                state.model_type = st.selectbox('Select Classification model to predict with.',
                                    class_model_opts, class_model_opts.index(state.model_type) if state.model_type in class_model_opts else 0)

                train_button = st.checkbox('Train')
                if train_button:

                    ### MODEL EVALUATION ###
                    fitting_load_state = st.text('Fitting and predicting...')
                    ypred, yscore, model = train_and_predict_bi_classification(xtrain, ytrain, xtest, state.model_type)
                    fitting_load_state.text('Fitting and predicting... done!')
                    state.model = model
                    st.header('Model Fit Diagnostics')
                    #download predictions
                    download_predictions(xtest, ypred, ytest)
                    download_model(state.model)

                    try:
                        eval_df, conf_matrix = evaluate_classification_model(ytest, ypred, yscore, state.model_type, state.model_method)
                        st.table(eval_df)
                    except Exception as e:
                        st.error(e)

                    try:
                        st.subheader('Confusion Matrix')
                        plot_conf_matrix(conf_matrix)
                    except Exception as e:
                        st.error(e)

                    try:
                        falsepos, truepos, roc_auc = get_roc_auc(ytest, yscore)
                        plot_multi_auc_roc(state.model_type, falsepos, truepos, roc_auc)
                    except Exception as e:
                        st.error(e)

                    try:
                        best_threshold, best_fscore = plot_prec_recall(ytest, yscore[:, 1])
                        st.write('Best Threshold=%f \nBest F-score=%f' % (best_threshold, best_fscore))
                    except Exception as e:
                        st.error(e)

                    if state.cv_button:
                        state.split_number = st.number_input('Number of splits:', state.split_number if state.split_number else 4)
                        cv_scores = get_cv_score(model, x, y, state.model_method, state.split_number)
                        plot_cv(cv_scores, state.model_method)
                        # cv_plot = plot_cv(cv_scores, state.model_method)
                        # st.plotly_chart(cv_plot, bbox_inches='tight')

                    return model, xtrain, xtest, ytrain, ytest, (ypred, yscore)

            elif state.model_method == 'Multi Class Classification':
                st.subheader('Multi Class Classification')

                if len(set(df[state.pred_var])) < 3:
                    st.error('Need 3 or more unique target values for multi class classification.')

                ## Modeling
                class_model_opts = list(Model.multi_classification_models.keys())
                state.model_type = st.selectbox('Select Classification model to predict with.',
                                                class_model_opts, class_model_opts.index(state.model_type) if state.model_type in class_model_opts else 0)
                train_button = st.checkbox('Train')
                if train_button:
                    ### MODEL EVALUATION ###
                    fitting_load_state = st.text('Fitting and predicting...')
                    ypred, yscore, model = train_and_predict_multi_classification(xtrain, ytrain, xtest, state.model_type)
                    fitting_load_state.text('Fitting and predicting... done!')
                    state.model = model
                    st.header('Model Fit Diagnostics')
                    #download predictions
                    download_predictions(xtest, ypred, ytest)
                    download_model(state.model)
                    try:
                        eval_df, conf_matrix = evaluate_multi_classification_model(ytest, ypred, yscore, state.model_type)
                        st.table(eval_df)

                    except Exception as e:
                        st.error(e)

                    try:
                        st.subheader('Confusion Matrix')
                        plot_conf_matrix(conf_matrix)
                    except Exception as e:
                        st.error(e)

                    try:
                        falsepos, truepos, roc_auc = get_roc_auc(ytest, yscore)
                        plot_multi_auc_roc(state.model_type, falsepos, truepos, roc_auc)
                    except Exception as e:
                        st.error(e)



                    if state.cv_button:
                        state.split_number = st.number_input('Number of splits:', state.split_number if state.split_number else 4)
                        cv_scores = get_cv_score(model, x, y, state.model_method, state.split_number)
                        plot_cv(cv_scores, state.model_method)
                        # cv_plot = plot_cv(cv_scores, state.model_method)
                        # st.pyplot(cv_plot, bbox_inches='tight')
                    return model, xtrain, xtest, ytrain, ytest, (ypred, yscore)

            elif state.model_method == 'Regression':
                st.subheader('Regression')
                reg_model_opts = list(Model.regression_models.keys())
                state.model_type = st.selectbox('Select regression model to predict with.',
                                            reg_model_opts, reg_model_opts.index(state.model_type) if state.model_type in reg_model_opts else 0)

                train_button = st.checkbox('Train')
                if train_button:
                    fitting_load_state = st.text('Fitting and predicting...')
                    ypred, model = train_and_predict_regression(xtrain, ytrain, xtest, state.model_type)
                    fitting_load_state.text('Fitting and predicting... done!')
                    state.model = model
                    ### MODEL EVALUATION ###
                    st.subheader('Model Fit Diagnostics')

                    #download predictions
                    download_predictions(xtest, ypred, ytest)
                    download_model(state.model)

                    # evaluation dataframe
                    try:
                        eval_df = evaluate_regression_model(ytest, ypred)
                        st.table(eval_df)
                    except Exception as e:
                        st.error(e)

                    try:
                        plot_regression_results(ytest, ypred, state.pred_var)
                    except Exception as e:
                        st.error(e)
                    if state.cv_button:
                        state.split_number = st.number_input('Number of splits:', state.split_number if state.split_number else 4)
                        cv_scores = get_cv_score(model, x, y, state.model_method, state.split_number)
                        plot_cv(cv_scores, state.model_method)
                        # cv_plot = plot_cv(cv_scores, state.model_method)
                        # st.pyplot(cv_plot, bbox_inches='tight')

                    return model, xtrain, xtest, ytrain, ytest, ypred

    else:
        pass


def explainability(model, xtrain, xtest, ytrain, ytest, ypred, df):

    st.header('Explainability')
    xcolumns = xtrain.columns

    state.explainer_select = st.selectbox('Select Explainability method to use:',
                                            ['Select an option',
                                             'Feature Importance',
                                             'Global Summary',
                                             'Local Summary',
                                             'PDP/Interaction Plot',
                                             ])

    if state.explainer_select == 'Feature Importance':
        plot_importance(model, state.model_type, xtrain)

    elif state.explainer_select == 'Global Summary':
        if state.model_method in ['Binary Classification']:
            if state.model_type != 'Logistic Regression':

                #calculate shap values
                explainer, shap_values_g = calculate_shap_values(model, state.model_type, state.xtest)

                #class_ = st.number_input('Select output class for explanation', min_value=min(set(ytest)), max_value=max(set(ytest)), value=int(0))
                try:
                    st.subheader('SHAP')
                    class_ = 1
                    shap_summary_classification(model, xtest, shap_values_g, state.model_type, class_, df)

                except:
                    st.error('SHAPley calculations ran into an error.')

                try:
                    st.subheader('ELI5')
                    eli5_plot = explain_eli5_global(state.model)
                    components.html(eli5_plot.data.replace("\n", "")) # Refer Streamlit Components API Docs (https://docs.streamlit.io/en/stable/develop_streamlit_components.html)
                    # st.markdown(
                    #     eli5_plot.data.replace("\n", ""), unsafe_allow_html=True
                    # )
                except:
                    st.error('ELI5 calculations ran into an error')
            else:
                st.error('SHAP does not work with Logistic regression, because it is a linear model.')
        elif state.model_method == 'Multi Class Classification':
            if state.model_type != 'Logistic Regression':
                #calculate shap values
                explainer, shap_values_g = calculate_shap_values(model, state.model_type, state.xtest)

                #class_ = st.number_input('Select output class for explanation', min_value=min(set(ytest)), max_value=max(set(ytest)), value=int(0))
                try:
                    st.subheader('SHAP')
                    class_ = st.selectbox('Class', list(set(ytest)))
                    shap_summary_classification(model, xtest, shap_values_g, state.model_type, class_, df)

                except:
                    st.error('SHAPley calculations ran into an error.')

                try:
                    st.subheader('ELI5')
                    eli5_plot = explain_eli5_global(state.model)
                    components.html(eli5_plot.data.replace("\n", "")) # Refer Streamlit Components API Docs (https://docs.streamlit.io/en/stable/develop_streamlit_components.html)
                    # st.markdown(
                    #     eli5_plot.data.replace("\n", ""), unsafe_allow_html=True
                    # )
                except:
                    st.error('ELI5 calculations ran into an error')
            else:
                st.error('SHAP does not work with Logistic regression, because it is a linear model.')
        elif state.model_method == 'Regression':
            if state.model_type != 'Linear Regression':

                try:
                    st.subheader('SHAP')
                    ### SHAP ###
                    # calculate global shap values
                    state.explainer, state.shap_values_g = calculate_shap_values(state.model, state.model_type, state.xtest)

                    ## summary plot
                    shap_summary_regression(state.model, state.xtest, state.shap_values_g, state.model_type, state.df)

                    ## download predictions
                    shap_results = download_shap_results(state.shap_values_g, state.ypred, state.ytest, state.xtest.columns)
                    shap_results_button = st.checkbox("Show SHAP values")
                    if shap_results_button:
                        st.write(shap_results)
                except:
                    st.error('SHAPley calculations ran into an error.')

                try:
                    st.subheader('ELI5')
                    ## ELI5
                    eli5_plot = explain_eli5_global(state.model)
                    print(type(eli5_plot.data))
                    components.html(eli5_plot.data.replace("\n", "")) # Refer Streamlit Components API Docs (https://docs.streamlit.io/en/stable/develop_streamlit_components.html)
                    # st.markdown(eli5.formatters.format_as_html(eli5_plot.data), unsafe_allow_html=True)
                    # st.markdown(eli5_plot.data.replace("\n", ""), unsafe_allow_html=True)
                except:
                    st.error('ELI5 calculations ran into an error.')

            else:
                st.error('SHAP cannot work with Linear regression, because it is a linear model. ')

    elif state.explainer_select == 'Local Summary':
        ### SHAP ###
        if state.model_method in ['Binary Classification', 'Multi Class Classification']:
            if state.model_type != 'Logistic Regression':

                #calculate shap values
                #explainer, shap_values_g = calculate_shap_values(model, state.model_type, state.xtest)


                ## single prediction
                st.subheader('Local Interpretation')
                individual = st.number_input('Select the desired record from the predicted set for detailed explanation:',
                                            min_value=min(range(len(xtest))),
                                            max_value=max(range(len(xtest))),
                                            value = 0)

                class_ = st.selectbox('Class', list(set(ytest)))
                shap_local_plot_classification(model, state.model_type, xtest, ytest, ypred[1], individual, class_)

            else:
                st.error('SHAP does not work with Logistic regression, because it is a linear model.')
        elif state.model_method == 'Regression':
            if state.model_type != 'Linear Regression':

                ##single prediction
                st.subheader('Local Interpretation')
                individual = st.number_input('Select the desired record from the predicted set for detailed explanation:',
                                            min_value=min(range(len(state.xtest))),
                                            max_value=max(range(len(state.xtest))),
                                            value = 0)
                shap_local_plots_regression(state.xtest, state.ytest, state.ypred, state.model, state.model_type, individual)

            else:
                st.error('SHAP cannot work with Linear regression, because it is a linear model. ')
    elif state.explainer_select == 'Dependence Plot':
        ### Variable Dependence plot
        if state.model_method in ['Binary Classification', 'Multi Class Classification']:
            if state.model_type != 'Logistic Regression':

                #calculate shap values
                explainer, shap_values_g = calculate_shap_values(model, state.model_type, state.xtest)


                ## single prediction
                st.subheader('Dependence Plot')
                individual = st.number_input('Select the desired record from the predicted set for detailed explanation:',
                                            min_value=min(range(len(xtest))),
                                            max_value=max(range(len(xtest))),
                                            value = 0)

                class_ = st.selectbox('Class', list(set(ytest)))
                shap_local_plot_classification(model, state.model_type, xtest, ytest, ypred[1], individual, class_)

            else:
                st.error('SHAP does not work with Logistic regression, because it is a linear model.')
        elif state.model_method == 'Regression':
            if state.model_type != 'Linear Regression':

                ##single prediction
                st.subheader('Local Interpretation')
                individual = st.number_input('Select the desired record from the predicted set for detailed explanation:',
                                            min_value=min(range(len(state.xtest))),
                                            max_value=max(range(len(state.xtest))),
                                            value = 0)
                shap_local_plots_regression(state.xtest, state.ytest, state.ypred, state.model, state.model_type, individual)

            else:
                st.error('SHAP cannot work with Linear regression, because it is a linear model. ')


    # elif state.explainer_select == 'ELI5':
    #     st.info('Needs work.')
    #     st.subheader('Global Explanation')
    #     eli5_plot = explain_eli5_global(state.model)
    #     st.markdown(
    #                 eli5_plot.data.replace("\n", ""), unsafe_allow_html=True
    #             )

    # elif state.explainer_select=='Tree Interpreter':
    #     st.header('Tree Interpreter')

    #     if state.model_method in ['Binary Classification', 'Multi Class Classification']:
    #         if state.model_type not in ['Logistic Regression', 'XGBoost Classifier']:
    #             predictions, biases, contributions = tree_interpret_predict(model, xtest)

    #             # choose prediction instance
    #             i = st.number_input('Select desired record from test set for detailed explanation:',
    #                         min_value = min(range(len(xtest))),
    #                         max_value = max(range(len(xtest))),
    #                         value = 0)

    #             pred_bias, ti_results = tree_interpret_bi_class(predictions, biases, contributions, i, xtest)

    #             st.write(pred_bias)
    #             st.subheader('Feature Contributions')
    #             show_results = st.checkbox("Show Tree Interpreter results dataframe")
    #             if show_results:
    #                 st.write(ti_results)

    #             plot_ti_classification(ti_results)
    #         else:
    #             st.error('Logistic Regression/XGBoost cannot work with Tree Interpreter')
    #     elif state.model_method == 'Regression':
    #         if state.model_type not in ['Linear Regression', 'XGBoost Regressor']:
    #             predictions, biases, contributions = tree_interpret_predict(model, xtest)

    #             # choose prediction instance
    #             i = st.number_input('Select desired record from test set for detailed explanation:',
    #                         min_value = min(range(len(xtest))),
    #                         max_value = max(range(len(xtest))),
    #                         value = 0)

    #             pred_bias, ti_results = tree_interpret_regression(predictions, biases, contributions, i, xtest)

    #             st.write('Actual', ytest.reset_index().drop('index', axis=1).iloc[i])
    #             st.write(pred_bias)
    #             st.subheader('Feature Contributions')
    #             show_results = st.checkbox("Show Tree Interpreter results dataframe")
    #             if show_results:
    #                 st.write(ti_results)

    #             plot_ti_regression(ti_results)
    #         else:
    #             st.error('Linear Regression/XGBoost cannot work with Tree Interpreter')

    # elif state.explainer_select == 'ICE Plot':
    #     st.header('ICE Plot')
    #     '''
    #     Observe how 1 or 2 features affect the prediction output.
    #     '''

    #     if state.model_type not in ['XGBoost Classifier']:
    #         num_select = st.selectbox('Select number of features in a plot', ['1', '2'])

    #         if num_select == '1':
    #             feature = st.selectbox('Select Feature to analyze', xcolumns)

    #             ice_df = ice_calc(xtest, model, feature)
    #             plot_ice(ice_df, feature)
    #         elif num_select == '2':
    #             cols_to_select = xcolumns.copy()
    #             feature1 = st.selectbox('Select 1st feature to analyze', cols_to_select)
    #             cols_to_select = cols_to_select.drop(feature1)
    #             feature2 = st.selectbox('Select 2nd feature to analyze', cols_to_select)

    #             ice_df = ice_calc(xtest, model, feature1)
    #             plot_ice_2(ice_df, feature1, feature2)
    #     else:
    #         st.error('XGBoost cannot work with ICE Plots')
    elif state.explainer_select == 'PDP/Interaction Plot':
        st.header('Partial Dependency Plot')
        pdp_plot_type = st.selectbox('Select Plot type',['Actual & Prediction Distribution',
                                      'Isolate',
                                      'Interaction'])

        if pdp_plot_type == 'Actual & Prediction Distribution':
            #st.subheader('Actual and Prediction Distribution Plots')
            '''
            Observe the distribution of the actual target value and the prediction value relating to a feature's values.
            '''
            feat_to_summarize = st.selectbox('Select Feature', xcolumns)
            distribution_summary(df, feat_to_summarize, df.drop(state.pred_var, axis=1), state.pred_var, model)

        elif pdp_plot_type == 'Isolate':
            #st.subheader('PDP Isolate Plot')
            '''
            See how a feature's values affect the model output. We can observe what type of trend the feature has
            relating to the model prediction output.
            '''
            feat_to_summarize = st.selectbox('Select Feature', xcolumns)
            plot_isolate(pdp_isolate(model, df, xcolumns, feat_to_summarize),
                            feat_to_summarize)

        elif pdp_plot_type == 'Interaction':
            #st.subheader('PDP Interaction Plot')
            '''
            See how two features interact with each other regarding the model's prediction output.
            '''
            cols_to_select = xcolumns.copy()
            feature1 = st.selectbox('Select 1st feature to analyze', cols_to_select)
            cols_to_select = cols_to_select.drop(feature1)
            feature2 = st.selectbox('Select 2nd feature to analyze', cols_to_select)
            plot_pdp_interaction(
                inter=pdp_interaction(model, df, xcolumns, feature1, feature2),
                feature1=feature1,
                feature2=feature2
            )
    # elif state.explainer_select == 'Correlation Graph':
    #     st.header('Correlation Graph')
    #     st.text('Network graph showing Pearson correlation.')
    #     st.text('The size of nodes indicate how many other features it is correlated with.')
    #     st.text('Width of edges indicate how much two features are correlated with each other.')

    #     corr_matrix = calculate_pearsons(df, state.pred_var)

    #     corr_direction = st.selectbox('Select to show positive or negative correlations', ['positive', 'negative'])
    #     if corr_direction == 'positive':
    #         corr_threshold = st.number_input('Select minimum threshold correlation level', min_value=0.0, max_value=1.0, value=0.5)
    #     elif corr_direction == 'negative':
    #         corr_threshold = st.number_input('Select minimum threshold correlation level', min_value=-1.0, max_value=0.0, value=-0.5)
    #     corr_network_load_state = st.text("Rendering correlation graph...")
    #     create_corr_network(corr_matrix, corr_direction, corr_threshold)
    #     corr_network_load_state.text("Rendering correlation graph... Done!")

# def clear_state():
#    """Clear all values stored in Session State"""
#     state.processed_df = None
#     state.model = None
#     state.xtrain = None
#     state.xtest = None
#     state.ytrain = None
#     state.ytest = None
#     state.ypred = None
#     state.pred_var = None
#     state.drop_cols = None
#     state.drop_button = None
#     state.encode_method_select = None
#     state.cols_to_encode = None
#     state.encode_button = None
#     state.model_method = None
#     state.model_type = None

def notes():
    st.header('Notes')
    '''
    \n
    When working with more than one uploaded dataset, you may need to refresh the application again to switch from
    one dataset to another.
    '''


#### MAIN ####
def main():
    # subscription_end_date = '2021-02-28'
    # now = datetime.now()
    # later = datetime.strptime(subscription_end_date, '%Y-%m-%d')
    if state.user_name == None:
            title = st.empty()
            logo = st.empty()
            text1 = st.empty()
            text2 = st.empty()
            usrname_placeholder = st.empty()
            pwd_placeholder = st.empty()
            submit_placeholder = st.empty()
            print('log in page initiated')

            title.title('Amruta XAI')
            logo.image(Image.open(LOGO_IMAGE), width=300)
            text1.text('Copyright Amruta Inc. 2021')
            text2.text('Beta/Test Version')
            state.usrname = usrname_placeholder.text_input("User Name", state.usrname if state.usrname else '')
            state.pwd = pwd_placeholder.text_input("Password", type="password", value=state.pwd if state.pwd else '')
            state.submit = submit_placeholder.button("Log In", state.submit)
            print('log in elements generated')

            if state.submit:
                print(state.submit)
                state.validation_status = validate(state.usrname, state.pwd)
                if state.validation_status == 'Access Granted':
                    # store input username to session state
                    state.user_name = state.usrname
                    print(state.user_name)

                    # empty login page elements
                    title.empty()
                    logo.empty()
                    text1.empty()
                    text2.empty()
                    usrname_placeholder.empty()
                    pwd_placeholder.empty()
                    submit_placeholder.empty()

                    # start main app
                    print('main app entered')
                    render_app()
                elif state.validation_status == 'Invalid username/password':
                    print('Invalid username/password')
                    st.error("Invalid username/password")
                elif state.validation_status == 'Subscription Ended':
                    print('Your subscription has ended. Please contact us to extend it.')
                    st.info("Your subscription has ended. Please contact us to extend it.")
                # elif:
                #     st.error("invalid credentials")
    else:
        render_app()


main()
