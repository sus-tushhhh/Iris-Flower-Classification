# Step 1: Load Important modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import streamlit as st
from sklearn.datasets import load_iris
import pickle
import altair as alt
import time
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(layout='wide', initial_sidebar_state='expanded')

# LOAD DATASET
data = load_iris()
df = pd.DataFrame(data['data'], 
                  columns = data['feature_names'])
df['target'] = data['target']
classes = data['target_names']

X = df.iloc[:,:-1]

# MODEL_LIST
all_model_name = ['Logistic Regression',
                 'Naive Bayes',"Decision Tree",
                 "Random Forest","SVM",
                 "KNN"]



all_models = []
for i in all_model_name:
    file_name = i.lower() +'.pkl'
    with open(f"ml_models/{file_name}", 'rb') as f:
        model = pickle.load(f)
        all_models.append(model)

# USER INPUT AND PAGE TITLE
st.title("ML Flower Classification Project")
# Image url
url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQF2roQNP1rPFtklA8xgZt76jyhj6x2BUjVe6gxwxJ53pI0_TYfQLRZh8oZ&s=10"
st.image(url)

# Show Dataframe sample
st.dataframe(df.sample(5))

# LEFT SIDE BAR for USER VALUE INPUT
st.sidebar.title("Select Iris Features")
st.sidebar.image(url)

user_input = []
for i in X:
    min_i = X[i].min()
    max_i = X[i].max()
    ans = float(st.sidebar.slider(f"Select value of {i}:", min_value = min_i, max_value = max_i))

    user_input.append(ans)

# USER INPUT SHOW
st.markdown("""
<h2> User Input Value</h2>
""",unsafe_allow_html=True)
st.write(user_input)

# MODEL PREDICTION
if st.button('Predict'):
    with st.spinner('Predicting'):
        time.sleep(2)
        counter = 0
        prediction_df = pd.DataFrame(columns=['Model Name', 'Prediction', 'Probability'])
        for model in all_models:
            ans = model.predict([user_input])[0]

            try:
                probability = model.predict_proba([user_input]).max()
            except Exception as e:
                probability=1

            class_ans = classes[ans]
            prediction_df.loc[counter+1] = [all_model_name[counter], class_ans.title(), probability]
            counter += 1
        
        st.header('Prediction : ')

        r, l = st.columns(2)
        with r: 
            st.dataframe(prediction_df)
        with l:
            with st.container(border=True):
                chart = (alt.Chart(prediction_df).mark_bar().encode(
                    x='Model Name',
                    y='Probability',
                    tooltip=['Model Name', 'Probability', 'Prediction']
                )).configure_axisX(labelAngle=0)
                st.altair_chart(chart, use_container_width=True)

        st.success(f'Final Prediction : {prediction_df['Prediction'].mode()[0]}')
    
    
footer = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #1A1C24;
    color: #f1f1f1;
    text-align: center;
    padding: 10px;
    font-size: 14px;
}
</style>
<div class="footer">
    <p>Made with ❤️ using <a href="https://streamlit.io/" target="_blank">Streamlit</a></p>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)

# All done