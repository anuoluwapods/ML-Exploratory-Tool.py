import streamlit as st
import pandas as pd
import numpy as np
import os
from openpyxl import load_workbook
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#Loading the data
def get_data_titanic():
    return pd.read_csv('tested.csv')
  
#configuration of the page
st.set_page_config(layout="wide")
st.title('Classification exploratory tool')

#load the data
df = get_data_titanic()
st.header('Original dataset')
st.write(df)
