import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (accuracy_score, recall_score, roc_auc_score, f1_score, 
                             precision_score, confusion_matrix, brier_score_loss, roc_curve)
                             
import shap
import joblib
import os

title = "Multimodal Clinical Decision Support for Post-Stroke Cognitive Impairment"

st.set_page_config(page_title=f"{title}", layout="wide", page_icon="🖥️")

# 加载归模型
try:
    st.write(os.path.exists("explainer.pkl"))
    lrmodel = joblib.load("LogisticRegression.pkl")
    lrexplainer = joblib.load("explainer.pkl")
    
    scaler = lrmodel.named_steps['scaler']
    clf = lrmodel.named_steps['clf']
  st.write(123)
except:
    st.error("**错误：**未找到 LogisticRegression.pkl 文件或 explainer.pkl 文件，请检查路径。")


st.markdown(f'''
    <h1 style="text-align: center; font-size: 26px; font-weight: bold; color: white; background: #3478CE; border-radius: 0.5rem; margin-bottom: 15px;">
        {title}
    </h1>''', unsafe_allow_html=True)

D = {'DTABR(global)': 3.72, 'DAR(frontal)': 4.92, 'Age': 63.0, 'A-MMD': 48.78, 'D-MFO': 3.16, 'FPN_beta': 0.369} # 初始值
selected_col = ['DTABR(global)', 'DAR(frontal)', 'Age', 'A-MMD', 'D-MFO', 'FPN_beta']
modelcol = ['DTABR(global)', 'A-MMD', 'D-MFO', 'FPN_beta', 'Age', 'DAR(frontal)']

data = {}
with st.form("inputform"):
    col = st.columns(3)
    for i, j in enumerate(selected_col):
        if i==2:
            data[j] = col[i%3].number_input(j, step=1, min_value=0, max_value=100, value=int(D[j]))
        elif i==5:
            data[j] = col[i%3].number_input(j, step=0.001, min_value=0.000, max_value=100.000, value=D[j])
        else:
            data[j] = col[i%3].number_input(j, step=0.01, min_value=0.00, max_value=100.00, value=D[j])
            
    c1 = st.columns(3)
    bt = c1[1].form_submit_button("**Start prediction**", use_container_width=True, type="primary")
    
if "predata" not in st.session_state:
    st.session_state.predata = data
else:
    pass


def prefun():
    pred_data = pd.DataFrame([st.session_state.predata])[modelcol]
    pred_data_df = pd.DataFrame(scaler.transform(pred_data), columns=pred_data.columns)
    
    with st.expander("**Current input**", True):
        st.dataframe(pred_data, hide_index=True, use_container_width=True)
    
    res = lrmodel.predict(pred_data)
    proba = round(float(lrmodel.predict_proba(pred_data)[0][1])*100, 2)
    
    if proba>=43.7:
        res = "PSCI"
    else:
        res = "PSN"
    
    with st.expander("**Prediction result**", True):
        st.info(f"The optimal diagnostic threshold for this model is 43.7%. A predicted probability of ≥ 43.7% indicates a high likelihood of PSCI. :gray[**Note: This model is intended for clinical reference only and should not replace a medical diagnosis**].")
        st.markdown(f'''
             <div style="text-align: center; font-size: 26px; color: black; margin-bottom: 5px; font-family: Times New Roman; border-bottom: 1px solid black;">
             Predicted Risk of PSCI: {proba}%
             </div>''', unsafe_allow_html=True)
            
        shap_values = lrexplainer.shap_values(pred_data_df)

        shap.force_plot(
            lrexplainer.expected_value,
            shap_values[0],
            pred_data.round(3),
            feature_names=pred_data.columns,
            matplotlib=True,
            show=False,
            link='logit'
        )

        plt.gca().grid(False)

        xlim = plt.gca().get_xlim()

        if xlim[0]<0.437<xlim[1]:
            plt.gca().vlines(0.437, 0, 0.2, color="red")
            plt.gca().text(0.437, 0.21, str(0.437), ha="center", va="top", color="red")
        
        col = st.columns([1, 6, 1])
        col[1].pyplot(plt.gcf(), use_container_width=True)
        
if bt:
    st.session_state.predata = data
    prefun()
else:

    prefun()


