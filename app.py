import streamlit as st
import pandas as pd
from pycaret.regression import setup, compare_models, pull, save_model, load_model

# Load the dataset if it exists
if st.sidebar.button("Reload Data"):
    df = pd.read_csv('dataset.csv', index_col=None)
else:
    df = pd.DataFrame()

with st.sidebar:
    st.title("Neural Nexus AutoML")
    choice = st.radio("Navigation", ["Upload", "Profiling", "Modelling", "Download"])
    st.info("Neural Nexus Automated AI platform allows you to create your ML pipelines all automated!")
    
if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file)
        df.to_csv('dataset.csv', index=False)
        st.success("Dataset uploaded successfully!")
        st.dataframe(df)

elif choice == "Profiling" and not df.empty:
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

elif choice == "Modelling" and not df.empty:
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

elif choice == "Download" and os.path.exists('best_model.pkl'):
    with open('best_model.pkl', 'rb') as f:
        st.download_button('Download Model', f, file_name="best_model.pkl")
else:
    st.warning("Please upload a dataset before proceeding.")
