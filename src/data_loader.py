import pandas as pd
import streamlit as st
from src.prediction import load_model as _load_model

@st.cache_data
def load_data():
    df = pd.read_csv("data/car_details.csv")
    df = df.drop_duplicates()
    df["brand"] = df["name"].str.split().str[0]
    df["brand"] = df["brand"].replace({"Land": "Land Rover", "OpelCorsa": "Opel"})
    df["car_age"] = 2020 - df["year"]
    return df

@st.cache_resource
def load_model():
    return _load_model()
