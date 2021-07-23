import streamlit as st
from PIL import Image # Required to show images
import math
# import matplotlib.pyplot as plt
# import seaborn as sns

import numpy    as np
import pandas   as pd
from sklearn import pipeline      # Pipeline
from sklearn import preprocessing # OrdinalEncoder, LabelEncoder
from sklearn import impute
from sklearn import compose
from sklearn import model_selection # train_test_split
from sklearn import metrics         # accuracy_score, balanced_accuracy_score, plot_confusion_matrix
from sklearn import set_config
from sklearn.model_selection import train_test_split


from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.ensemble      import ExtraTreesClassifier
from sklearn.ensemble      import AdaBoostClassifier
from sklearn.ensemble      import GradientBoostingClassifier
from sklearn.experimental  import enable_hist_gradient_boosting # Necesary for HistGradientBoostingClassifier
from sklearn.ensemble      import HistGradientBoostingClassifier
from xgboost               import XGBClassifier


    
    



header_container = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
model_training = st.beta_container()


with header_container:
    st.title('Welcome to my firt streamlit project!')




with dataset:
    st.header('Crypto app')




with features:
    st.header('Features')
    
    
    
    
with model_training:
    st.header('Time to train the model')