# ###### TREE INTERPRETER ######
# from treeinterpreter import treeinterpreter as ti
# import pandas as pd
# import streamlit as st
# import plotly.express as px

# #@st.cache
# def tree_interpret_predict(model, x):
#     predictions, biases, contributions = ti.predict(model, x)
#     return predictions, biases, contributions

# ## Plot

# def tree_interpret_bi_class(predictions, biases, contributions, i, x):
#     ''' Tree Interpreter for Binary Classification '''

#     features=[]
#     contribution=[]
#     abscontribution=[]
#     for c, feature in list(zip(contributions[i], x.columns)):
#         features.append(feature)
#         contribution.append(c)
#         abscontribution.append(abs(c[1] if c[1]>c[0] else abs(c[0])))
#         #contribution1.append(abs(c[1]))
#         #print(feature, c)

#     pred_bias = pd.DataFrame({
#                     '0':[predictions[i][0], biases[i][0]],
#                     '1':[predictions[i][1], biases[i][1]]
#                 }, index=['Prediction', 'Bias'])
#     ti_results = pd.DataFrame({'features':features,
#         'contribution 0':list(list(zip(*contribution))[0]),
#         'contribution 1':list(list(zip(*contribution))[1]),
#         'abs_contributions':abscontribution}).sort_values(by='abs_contributions', ascending=False)[:20]

#     return pred_bias, ti_results


# def tree_interpret_regression(predictions, biases, contributions, i, x):
#     ''' Tree Interpreter for Binary Classification '''

#     features=[]
#     contribution=[]
#     abscontribution=[]
#     for c, feature in list(zip(contributions[i], x.columns)):
#         features.append(feature)
#         contribution.append(c)
#         abscontribution.append(abs(c))

#     pred_bias = pd.DataFrame({
#                     'Value':[predictions[i][0], biases[i]]
#                 }, index=['Prediction', 'Bias'])

#     ti_results = pd.DataFrame({'features':features,
#                                 'contributions':contribution,
#                                 'abs_contributions':abscontribution}).sort_values(by='abs_contributions', ascending=False)[:20]

#     return pred_bias, ti_results


# def plot_ti_regression(ti_results):
#     fig = px.bar(ti_results, x='contributions', y='features', orientation='h')
#     st.plotly_chart(fig, bbox_inches='tight')
#     return

# def plot_ti_classification(ti_results):
#     fig = px.bar(ti_results, x='contribution 1', y='features', orientation='h')
#     st.plotly_chart(fig, bbox_inches='tight')
#     return