from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import pandas as pd
import os 

# Function to generate basic summary statistics
def generate_summary(df):
    summary = df.describe()
    return summary

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv')
else:
    df = pd.DataFrame()

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Neural Nexus AutoML")
    choice = st.radio("Navigation", ["Upload", "Modelling", "Download"])
    st.info("Neural Nexus Automated AI platform lets allows you to create your ML pipelines all automated!")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file)
        df.to_csv('dataset.csv', index=False)
        st.success("Dataset uploaded successfully!")
        st.dataframe(df)
        st.write("Summary Statistics:")
        st.write(generate_summary(df))

if choice == "Modelling" and not df.empty:
    st.title("Model Training")
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Train'):
        setup(df, target=chosen_target)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model.pkl')
        st.success("Model trained successfully!")

if choice == "Download" and os.path.exists('best_model.pkl'):
    with open('best_model.pkl', 'rb') as f:
        st.download_button('Download Model', f, file_name="best_model.pkl")
