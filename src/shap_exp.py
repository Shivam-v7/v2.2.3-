# AMRUTA OXAI V2
# Copyright Amruta Inc. 2021
# Author: Dishit Kotecha

from src.modeling import *
import shap
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import base64

###### SHAP ######

st.set_option('deprecation.showPyplotGlobalUse', False)


@st.cache(allow_output_mutation=True, hash_funcs={XGBClassifier:id, XGBRegressor:id})
def shap_explainability(model, xtest, model_type):
    if model_type in Model.tree_based_models:
        print('running tree explainer')
        explainer = shap.TreeExplainer(model)
    else:
        print('running kernel explainer')
        with st.spinner('Running Explainer, this will take longer for non-tree based models.'):
            explainer = shap.KernelExplainer(model.predict, xtest)

    return explainer


def calculate_shap_values(model, model_type, x_features):
    explainer = shap_explainability(model, x_features, model_type)
    return explainer, explainer.shap_values(x_features)


def shap_summary_regression(model, x, shap_values_g, model_type, df):
    if model_type not in Model.tree_based_models:
        st.warning('Please be warned that SHAP calculations for non-tree based\
            algorithms will take significant time to perform.')

    plt.title('Assessing feature importance with SHAP values')
    shap.summary_plot(shap_values_g, x)
    st.pyplot(bbox_inches='tight')


def shap_summary_classification(model, x, shap_values, model_type, class_, df):
    if model_type not in Model.tree_based_models:
        st.warning('Please be warned that SHAP calculations for non-tree based\
            algorithms will take significant time to perform.')

    plt.title('Assessing feature importance with SHAP values')
    shap.summary_plot(shap_values, x, plot_type='bar', show=False)
    st.pyplot(bbox_inches='tight')
    plt.clf()

    plt.title('Summary of Effects of All Features on Class %d'%(class_))
    if 'XGBoost' in model_type or 'LGBM' in model_type:
        shap.summary_plot(shap_values, x)
    else:
        shap.summary_plot(shap_values[class_], x)
    st.pyplot(bbox_inches='tight')
    plt.clf()


from scipy.special import expit  # Importing the logit function for the base value transformation


def shap_transform_scale(shap_values, expected_value, model_prediction):

    # Compute the transformed base value, which consists in applying the logit function to the base value
    expected_value_transformed = expit(expected_value)

    # Computing the original_explanation_distance to construct the distance_coefficient later on
    # print("sum shap_values", sum(shap_values))
    original_explanation_distance = sum(shap_values)

    # Computing the distance between the model_prediction and the transformed base_value
    # distance_to_explain = abs(model_prediction - expected_value_transformed)
    # print('model prediction values', model_prediction)
    distance_to_explain = model_prediction - expected_value_transformed

    # The distance_coefficient is the ratio between both distances which will be used later on
    # print("org_exp_dist", original_explanation_distance.shape)
    # print("dist_to_exp", distance_to_explain.shape)
    distance_coefficient = original_explanation_distance / distance_to_explain

    # Transforming the original shapley values to the new scale
    shap_values_transformed = shap_values / distance_coefficient

    return shap_values_transformed, expected_value_transformed


def shap_local_plots_regression(x, y, ypred, model, model_type, individual):

    predicted = np.array(ypred)[individual]
    actual = np.array(y)[individual]
    st.write('Actual = ', round(actual, 3))
    st.write('Predicted = ', round(predicted, 3))

    # calculate shap values
    if 'XGBoost' in model_type:
        explainer, shap_values = calculate_shap_values(model, model_type, x)
        shap.waterfall_plot(explainer.expected_value, shap_values[0], x.iloc[individual, :])
        absvals = np.abs(shap_values).mean(0)
        shap_values = shap_values[0]

    elif 'LGBM' in model_type:
        explainer, shap_values = calculate_shap_values(model, model_type, x)
        # print(shap_values[0])
        shap.waterfall_plot(explainer.expected_value, shap_values[0], x.iloc[individual, :])
        absvals = np.abs(shap_values).mean(0)
        shap_values = shap_values[0]
    else:
        explainer, shap_values = calculate_shap_values(model, model_type, x)
        shap.waterfall_plot(explainer.expected_value, shap_values[individual], x.iloc[individual, :])
        absvals = np.abs(shap_values) # .mean(0)
        # print(shap_values)

    st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0)

    st.subheader('Features ordered by SHAP Importance')
    st.write('Base Value:', round(explainer.expected_value, 3))
    st.text('The value that would be predicted if we did not know any features for the current output.')
    st.write('Output Value:', round(explainer.expected_value+shap_values[individual].sum(), 3))
    try:
        feature_importance = pd.DataFrame({'Feature':x.columns.values,
                                           'SHAP Feature Importance': shap_values,
                                           'abs_values': absvals})
    except:
        feature_importance = pd.DataFrame({'Feature': x.columns.values,
                                           'SHAP Feature Importance': shap_values[individual],
                                           'abs_values': absvals[individual]})
    feature_importance.sort_values(by=['abs_values'], ascending=False, inplace=True)
    feature_importance = feature_importance[['Feature', 'SHAP Feature Importance']]
    inhibs = feature_importance[feature_importance['SHAP Feature Importance'] < 0]
    enhances = feature_importance[feature_importance['SHAP Feature Importance'] > 0]

    # output bar plots
    st.write(feature_importance.head(15))
    st.write('Inhibitors')
    st.write(inhibs)
    st.write('Enhancers')
    st.write(enhances)

    return


def shap_local_plot_classification(model, model_type, x, y, yscore, individual, class_=1):
    # force plot
    st.write("Predicting Class %d"%(class_))
    # single prediction
    predicted = np.array(yscore)[individual][class_]
    actual = np.array(y)[individual]
    st.write('Actual = ' + str(round(actual, 3)))

    st.write('Predicted = '+str(round(predicted, 3)))
    explainer, shap_values = calculate_shap_values(model, model_type, x)

    if 'XGBoost' in model_type:
        # print(shap_values[0])
        # print(explainer.expected_value)
        try:
            exp_value = explainer.expected_value[class_]
            try:
                shap.waterfall_plot(exp_value, shap_values[class_][individual, :], x.iloc[individual, :])
            except:
                shap.force_plot(exp_value, shap_values[class_][individual, :], x.iloc[individual, :])
        except:
            # exp_value = explainer.expected_value
            shap_values[0], exp_value = shap_transform_scale(shap_values[0], explainer.expected_value, predicted)
            try:
                shap.waterfall_plot(exp_value[individual], shap_values[individual], x.iloc[individual])
            except:
                shap.force_plot(exp_value, shap_values[individual], x.iloc[individual], matplotlib=True)
    elif 'LGBM' in model_type:
        exp_value = explainer.expected_value[class_]
        try:
            shap.waterfall_plot(exp_value, shap_values[class_][individual, :], x.iloc[individual, :])
        except:
            shap.force_plot(exp_value, shap_values[class_][individual, :], x.iloc[individual, :])

    else:
        exp_value = explainer.expected_value[class_]
        shap.waterfall_plot(exp_value, shap_values[class_][individual, :], x.iloc[individual, :])
    st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0)
    plt.clf()

    st.subheader('Features ordered by SHAP Importance')
    if 'XGBoost' in model_type:
        st.write('Base Value:', round(exp_value, 3))
        st.text('The value that would be predicted if we did not know any features for the current output.')
        st.write('Output Value:', round(exp_value+shap_values[0].sum(), 3))
        # absvals= np.abs(shap_values[individual])#.mean(0)
        try:
            absvals = np.abs(shap_values)  # .mean(0)
            feature_importance = pd.DataFrame({'Feature': x.columns.values,
                                           'SHAP Feature Importance': shap_values[0],
                                           'abs_values': absvals[0]})
        except:
            absvals = np.abs(shap_values[individual])  # .mean(0)
            feature_importance = pd.DataFrame({'Feature': x.columns.values,
                                               'SHAP Feature Importance': shap_values[class_][individual, :],
                                               'abs_values': absvals[individual]})
    elif 'LGBM' in model_type:
        st.write('Base Value:', round(exp_value, 3))
        st.text('The value that would be predicted if we did not know any features for the current output.')
        st.write('Output Value:', round(exp_value+shap_values[class_][individual,:].sum(), 3))
        absvals= np.abs(shap_values[class_][individual,:])#.mean(0)
        feature_importance = pd.DataFrame({'Feature':x.columns,
                                        'SHAP Feature Importance':shap_values[class_][individual,:],
                                        'abs_values':absvals})
    else:
        st.write('Base Value:', round(explainer.expected_value[class_], 3))
        st.text('The value that would be predicted if we did not know any features for the current output.')
        st.write('Output Value:', round(explainer.expected_value[class_]+shap_values[class_].sum(), 3))
        absvals= np.abs(shap_values[class_])#.mean(0)
        # print("abs", absvals[class_].shape)
        # print("x", x.columns.values.shape)
        # print("shap-class", len(shap_values[class_][individual, :]))
        feature_importance = pd.DataFrame({'Feature':x.columns.values,
                                            'SHAP Feature Importance':shap_values[class_][individual, :],
                                            'abs_values':absvals[class_]})
    feature_importance.sort_values(by=['abs_values'],ascending=False,inplace=True)
    feature_importance = feature_importance[['Feature', 'SHAP Feature Importance']]
    inhibs = feature_importance[feature_importance['SHAP Feature Importance']<0]
    enhances = feature_importance[feature_importance['SHAP Feature Importance']>0]
    st.write(feature_importance.head(10))
    st.write('Inhibitors')
    st.write(inhibs)
    st.write('Enhancers')
    st.write(enhances)

def shap_dependence_plots_regression(feature1, feature2, x, shap_values_g):

    try:
        shap.dependence_plot(ind=feature1, interaction_index=feature2, shap_values=shap_values_g, features=x)
        st.pyplot(bbox_inches='tight')
    except:
        st.error('Dependence plot cannot be outputted.')
        pass

def shap_dependence_plot_classification(model_type, feature1, feature2, x, shap_values_g, class_):
    #dependence plots

    if 'XGBoost' in model_type:
        shap.dependence_plot(ind=feature1, interaction_index=feature2, shap_values=shap_values_g, features=x)
    else:
        shap.dependence_plot(ind=feature1, interaction_index=feature2, shap_values=shap_values_g[class_], features=x)
    st.pyplot(bbox_inches='tight')

def show_shap_dependence_tabular_class(feature1, feature2, x, shap_values, feature_names, model_type, class_):
    if 'XGBoost' in model_type:
        if feature2 == "auto":
            feature2 = approximate_interactions(feature1, shap_values, x)[0]
            feature2 = feature_names[feature2]
            feature2_ind = convert_name(feature2, shap_values, feature_names)
        else:
            feature2_ind = convert_name(feature2, shap_values, feature_names)

        feature1_ind = convert_name(feature1, shap_values, feature_names)
        shap_values_1 = shap_values[:,feature1_ind]
        shap_values_2 = shap_values[:,feature2_ind]

        return pd.DataFrame({feature1+'_SHAP':shap_values_1,
                            feature1:x.iloc[:, feature1_ind],
                            feature2+'_SHAP':shap_values_2,
                            feature2: x.iloc[:, feature2_ind]
                            })
    else:
        if feature2 == "auto":
            feature2 = approximate_interactions(feature1, shap_values[class_], x)[0]
            feature2 = feature_names[feature2]
            feature2_ind = convert_name(feature2, shap_values[class_], feature_names)
        else:
            feature2_ind = convert_name(feature2, shap_values[class_], feature_names)

        feature1_ind = convert_name(feature1, shap_values[class_], feature_names)
        shap_values_1 = shap_values[class_][:,feature1_ind]
        shap_values_2 = shap_values[class_][:,feature2_ind]

        return pd.DataFrame({feature1+'_SHAP':shap_values_1,
                            feature1:x.iloc[:, feature1_ind],
                            feature2+'_SHAP':shap_values_2,
                            feature2: x.iloc[:, feature2_ind]
                            })


def show_shap_dependence_tabular_reg(feature1, feature2, x, shap_values, feature_names):

    if feature2 == "auto":
        feature2 = approximate_interactions(feature1, shap_values, x)[0]
        feature2 = feature_names[feature2]
        feature2_ind = convert_name(feature2, shap_values, feature_names)
    else:
        feature2_ind = convert_name(feature2, shap_values, feature_names)

    feature1_ind = convert_name(feature1, shap_values, feature_names)
    shap_values_1 = shap_values[:,feature1_ind]
    shap_values_2 = shap_values[:,feature2_ind]

    return pd.DataFrame({feature1+'_SHAP':shap_values_1,
                         feature1:x.iloc[:, feature1_ind],
                        feature2+'_SHAP':shap_values_2,
                        feature2: x.iloc[:, feature2_ind]
                        })

def download_shap_dependence_tabular(dependence_tabular):
    csv = dependence_tabular.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="shap_dependence_results.csv">Download SHAP Dependence Plot Results CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)

def download_shap_results(shap_values, predictions, actuals, columns):
    output_df = pd.DataFrame(shap_values, columns = columns)
    output_df['Actual'] = list(actuals)
    output_df['Prediction'] = list(predictions)
    csv = output_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="shap_results.csv">Download Test Set SHAP Values CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)
    return output_df

def convert_name(ind, shap_values, feature_names):
        if type(ind) == str:
            nzinds = np.where(np.array(feature_names) == ind)[0]
            if len(nzinds) == 0:
                # we allow rank based indexing using the format "rank(int)"
                if ind.startswith("rank("):
                    return np.argsort(-np.abs(shap_values).mean(0))[int(ind[5:-1])]

                # we allow the sum of all the SHAP values to be specified with "sum()"
                # assuming here that the calling method can deal with this case
                elif ind == "sum()":
                    return "sum()"
                else:
                    raise ValueError("Could not find feature named: " + ind)
                    return None
            else:
                return nzinds[0]
        else:
            return ind

def approximate_interactions(index, shap_values, X, feature_names=None):
    """ Order other features by how much interaction they seem to have with the feature at the given index.
    This just bins the SHAP values for a feature along that feature's value. For true Shapley interaction
    index values for SHAP see the interaction_contribs option implemented in XGBoost.
    """

    # convert from DataFrames if we got any
    if str(type(X)).endswith("'pandas.core.frame.DataFrame'>"):
        if feature_names is None:
            feature_names = X.columns
        X = X.values

    index = convert_name(index, shap_values, feature_names)

    if X.shape[0] > 10000:
        a = np.arange(X.shape[0])
        np.random.shuffle(a)
        inds = a[:10000]
    else:
        inds = np.arange(X.shape[0])

    x = X[inds, index]
    srt = np.argsort(x)
    shap_ref = shap_values[inds, index]
    shap_ref = shap_ref[srt]
    inc = max(min(int(len(x) / 10.0), 50), 1)
    interactions = []
    for i in range(X.shape[1]):
        encoded_val_other = encode_array_if_needed(X[inds, i][srt], dtype=np.float)

        val_other = encoded_val_other
        v = 0.0
        if not (i == index or np.sum(np.abs(val_other)) < 1e-8):
            for j in range(0, len(x), inc):
                if np.std(val_other[j:j + inc]) > 0 and np.std(shap_ref[j:j + inc]) > 0:
                    v += abs(np.corrcoef(shap_ref[j:j + inc], val_other[j:j + inc])[0, 1])
        val_v = v

        val_other = np.isnan(encoded_val_other)
        v = 0.0
        if not (i == index or np.sum(np.abs(val_other)) < 1e-8):
            for j in range(0, len(x), inc):
                if np.std(val_other[j:j + inc]) > 0 and np.std(shap_ref[j:j + inc]) > 0:
                    v += abs(np.corrcoef(shap_ref[j:j + inc], val_other[j:j + inc])[0, 1])
        nan_v = v

        interactions.append(max(val_v, nan_v))

    return np.argsort(-np.abs(interactions))

def encode_array_if_needed(arr, dtype=np.float64):
    try:
        return arr.astype(dtype)
    except ValueError:
        unique_values = np.unique(arr)
        encoding_dict = {string: index for index, string in enumerate(unique_values)}
        encoded_array = np.array([encoding_dict[string] for string in arr], dtype=dtype)
        return encoded_array