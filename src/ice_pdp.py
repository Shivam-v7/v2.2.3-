# AMRUTA OXAI V2
# Copyright Amruta Inc. 2021
# Author: Dishit Kotecha

### ICE AND PDP ###
from matplotlib.cm import PuOr, ScalarMappable
# from pycebox.ice import ice, ice_plot
from pdpbox import pdp, info_plots
import matplotlib.pyplot as plt
import streamlit as st

###### ICE PLOTS ######
## TODO: fix feature_names mismatch error for xgboost
# def ice_calc(x, model, feature):
#     ice_df = ice(x, feature, model.predict)
#     return ice_df


# def plot_ice(ice_df, feature):
#     ice_plot(ice_df, c='dimgray')
#     plt.ylabel('Prediction')
#     plt.xlabel(feature)
#     st.pyplot(bbox_inches='tight')
#     #feat_vals = ice_df.columns.get_level_values(feature).

# def plot_ice_2(ice_df, feature1, feature2):
#     ice_plot(ice_df, color_by=feature2, cmap=PuOr)
#     plt.ylabel('Prediction')
#     plt.xlabel(feature1)
#     #color map bar
#     feat2_vals = ice_df.columns.get_level_values(feature2).values
#     sm = ScalarMappable(cmap=PuOr, norm=plt.Normalize(vmin=feat2_vals.min(), vmax=feat2_vals.max()))
#     plt.colorbar(sm, label=feature2)
#     st.pyplot(bbox_inches='tight')

###### PDP PLOTS ######

def distribution_summary(df, feature, x, target, model):
    #target plot
    fig1, axes1, summary_df1 = info_plots.target_plot(
        df=df, feature=feature, feature_name=feature, target=target
    )
    st.pyplot(fig1, bbox_inches='tight')
    show_summary1 = st.checkbox('Show summary dataframe (actuals)')
    if show_summary1:
        st.write(summary_df1)

    #prediction plot
    fig2, axes2, summary_df2 = info_plots.actual_plot(
        model=model, X=x, feature=feature, feature_name=feature, predict_kwds={}
    )
    st.pyplot(fig2, bbox_inches='tight')
    show_summary2 = st.checkbox('Show summary dataframe (predictions)')
    if show_summary2:
        st.write(summary_df2)

#@st.cache
def pdp_isolate(model, df, xcolumns, feature):
    pdp_values = pdp.pdp_isolate(
        model=model, dataset=df, model_features=xcolumns, feature=feature
    )
    return pdp_values

def plot_isolate(pdp_values, feature):
    fig, axes = pdp.pdp_plot(pdp_values, feature, plot_lines=True, plot_pts_dist=True)
    st.pyplot(fig, bbox_inches='tight')

def pdp_interaction(model, df, model_features, feature1, feature2):
    inter = pdp.pdp_interact(model, df, model_features, features=[feature1, feature2])
    return inter

def plot_pdp_interaction(inter, feature1, feature2):
    fig, axes = pdp.pdp_interact_plot(
        inter, [feature1, feature2], x_quantile=True, plot_type='grid', plot_pdp=False
    )
    st.pyplot(fig, bbox_inches='tight')