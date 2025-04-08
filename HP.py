import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载模型
model = joblib.load('Model.pkl')

# Streamlit的用户界面
st.title("Biochar adsorption prediction platform (Heavy metals)")

# C(%): 数值输入
C = st.number_input("C(%):", min_value=8.47, max_value=88.30, value=63.33)

# H(%): 数值输入
H = st.number_input("H(%):", min_value=0.51, max_value=6.31, value=2.72)

# O(%): 数值输入
O = st.number_input("O(%):", min_value=0.30, max_value=27.99, value=15.39)

# N(%): 数值输入
N = st.number_input("N(%):", min_value=0.22, max_value=25.60, value=2.13)

# (O+N)/C: 数值输入
ONC = st.number_input("(O+N)/C:", min_value=0.01, max_value=0.85, value=0.27)

# O/C: 数值输入
OC = st.number_input("O/C:", min_value=0.00, max_value=0.77, value=0.24)

# H/C: 数值输入
HC = st.number_input("H/C:", min_value=0.02, max_value=0.70, value=0.15)

# Ash(%): 数值输入
Ash = st.number_input("Ash(%):", min_value=2.75, max_value=81.70, value=18.85)

# SSA(m²/g): 数值输入
SSA = st.number_input("SSA(m²/g):", min_value=0.73, max_value=1224.00, value=84.63)

# Pore volume(cm³/g): 数值输入
Pore_volume = st.number_input("Pore volume(cm³/g):", min_value=0.01, max_value=491.22, value=108.20)

# HMC(mg/L): 数值输入
HMC = st.number_input("HMC(mg/L):", min_value=1.00, max_value=500.00, value=129.80)

# Stirring speed(rpm): 数值输入
Stirring_speed = st.number_input("Stirring speed(rpm):", min_value=140.00, max_value=200.00, value=160.00)

# Volume (L): 数值输入
Volume = st.number_input("Volume(L):", min_value=0.02, max_value=0.25, value=0.05)

# Biochar concentration(g/L): 数值输入
Biochar_concentration = st.number_input("Biochar concentration(g/L):", min_value=0.01, max_value=20.00, value=3.00)

# Adsorption temperature(℃): 数值输入
Adsorption_temperature = st.number_input("Adsorption temperature(℃):", min_value=18.00, max_value=25.00, value=20.00)

# Adsorption time(min): 数值输入
Adsorption_time = st.number_input("Adsorption time(min):", min_value=0.00, max_value=4760.00, value=766.48)

# 处理输入并进行预测
feature_values = [C,H,O,N,ONC,OC,HC,Ash, SSA, Pore_volume, HMC,Stirring_speed,Volume,Biochar_concentration,Adsorption_temperature,Adsorption_time]
features = np.array([feature_values])

# 特征名称列表
feature_names = ["C", "H", "O", "N", "(O+N)/C", "O/C", "H/C", "Ash", "SSA", "Pore volume", "HMC", "Stirring speed", "Volume", "Biochar concentration", "Adsorption temperature", "Adsorption time"]

if st.button("Predict"):
    # 预测类别和概率
    predicted_proba = model.predict(features)

    # 显示预测结果
    st.write(f"**Adsorption capacity:** {predicted_proba}(mg/g)")
    
    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame(features, columns=feature_names))
    
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

    st.image("shap_force_plot.png")
