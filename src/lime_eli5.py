### LIME with ELI5
import eli5
import streamlit as st
import pandas as pd
import numpy as np


def explain_eli5_global(model):
    return eli5.show_weights(model)


def explain_local_eli5(model, ind, features):
    return eli5.show_prediction(model, ind, feature_names=features, show_feature_values=True)