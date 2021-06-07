# AMRUTA OXAI V2
# Copyright Amruta Inc. 2021
# Author: Dishit Kotecha

###### CORRELATION GRAPH ######
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

@st.cache
def calc_corr(df):
    return df.corr()

@st.cache
def calculate_pearsons(df, target, write_gdf=False):

    """ Encodes categorical variables, calculates Pearson correlations,
        and calls wite_gdf.
        Limit to 50 features.
    """

    # read csv and keep inputs in X list
    frame = df
    frame = frame.drop(target, axis=1)#.iloc[:, :num_features]

    # collect names of variables
    # to attempt to encode
    try_name_list = [name for name, type_ in frame.dtypes.items()
                     if type_ == 'object']

    print('Encoding categorical columns ...')

    # handle unary
    # don't encode unary categorical columns
    unary_list = [name for name in try_name_list if
                  len(frame[name].unique()) == 1]

    if len(unary_list) > 0:
        frame = frame.drop(unary_list, axis=1)
        try_name_list = list(set(try_name_list) - set(unary_list))

    # encode binary
    # don't create perfectly, negatively correlated encoded columns
    binary_list = [name for name in try_name_list if
                   len(frame[name].unique()) == 2]

    if len(binary_list) > 0:
        dummies = pd.get_dummies(frame[binary_list], dummy_na=True,
                                 drop_first=True)
        frame = frame.drop(binary_list, axis=1)
        frame = pd.concat([frame, dummies], axis=1)
        try_name_list = list(set(try_name_list) - set(binary_list))

    # encode nominal
    nominal_list = [name for name in try_name_list if
                    len(frame[name].unique()) <=
                    NUM_LEVELS_THRESHOLD and
                    len(frame[name].unique()) > 2]

    if len(nominal_list) > 0:
        dummies = pd.get_dummies(frame[nominal_list], dummy_na=True)
        frame = frame.drop(nominal_list, axis=1)
        frame = pd.concat([frame, dummies], axis=1)

    print('Done.')

    # calculate Pearson correlations
    print('Calculating Pearson correlations ...')
    corr_frame = calc_corr(frame).applymap(round_4)
    print('Done.')


    # write gdf
    if write_gdf:
        print('Writing GDF file to %s ...' % write_gdf(corr_frame))
        print('Done.')

    return corr_frame

def round_4(x):
    return round(x, 4)

def create_corr_network(corr_matrix, corr_direction='positive', min_correlation=0.2):

    ##create graph from corr matrix
    features = corr_matrix.index.values
    G = nx.from_numpy_matrix(np.asmatrix(corr_matrix))
    G = nx.relabel_nodes(G, lambda x: features[x])
    G.edges(data=True)

    ##Checks all the edges and removes some based on corr_direction
    for node1, node2, weight in list(G.edges(data=True)):
        ##if we only want to see the positive correlations we then delete the edges with weight smaller than 0
        if corr_direction == "positive":
            ####it adds a minimum value for correlation.
            ####If correlation weaker than the min, then it deletes the edge
            if weight["weight"] <0 or weight["weight"] < min_correlation:
                G.remove_edge(node1, node2)
        ##this part runs if the corr_direction is negative and removes edges with weights equal or largen than 0
        elif corr_direction == "negative":
            ####it adds a minimum value for correlation.
            ####If correlation weaker than the min, then it deletes the edge
            if weight["weight"] >=0 or weight["weight"] > min_correlation:
                G.remove_edge(node1, node2)

        else:
            raise ValueError("'corr_direction' only takes 'positive' or 'negative' argument")


    #creates a list for edges and for the weights
    edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())

    ### increases the value of weights, so that they are more visible in the graph
    weights = tuple([(1+abs(x))**2 for x in weights])

    #####calculates the degree of each node
    d = dict(nx.degree(G))
    #####creates list of nodes and a list their degrees that will be used later for their sizes
    nodelist, node_sizes = zip(*d.items())

    #positions
    positions=nx.circular_layout(G)

    #Figure size
    plt.figure(figsize=(15,15))

    #draws nodes
    nx.draw_networkx_nodes(G,positions,node_color='#DA70D6',nodelist=nodelist,
                           #####the node size will be now based on its degree
                           node_size=tuple([x**3 for x in node_sizes]),alpha=0.8)

    #Styling for labels
    nx.draw_networkx_labels(G, positions, font_size=8,
                            font_family='sans-serif')

    #Labeling edges
    nx.draw_networkx_edge_labels(G, positions, edge_labels=nx.get_edge_attributes(G, 'weight'),font_size=8, font_family='sans-serif')

    ###edge colors based on weight direction
    if corr_direction == "positive":
        edge_colour = plt.cm.GnBu
    elif corr_direction == 'negative':
        edge_colour = plt.cm.PuRd
    else:
        raise ValueError("'corr_direction' only takes 'positive' or 'negative' argument")

    #draws the edges
    nx.draw_networkx_edges(G, positions, edge_list=edges,style='solid',
                          ###adds width=weights and edge_color = weights
                          ###so that edges are based on the weight parameter
                          ###edge_cmap is for the color scale based on the weight
                          ### edge_vmin and edge_vmax assign the min and max weights for the width
                          width=weights, edge_color = weights, edge_cmap = edge_colour,
                          edge_vmin = min(weights), edge_vmax=max(weights))

    # displays the graph without axis
    plt.axis('off')
    #saves image
    #plt.savefig("" + corr_direction + ".png", format="PNG")
    #plt.show()

    st.pyplot(bbox_inches='tight')