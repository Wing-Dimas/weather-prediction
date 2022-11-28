import streamlit as st
import pandas as pd

def home():
    st.title("Weather Prediction")
    st.write("Website ini bertujuan untuk memprediksi cuaca yang akan terjadi, dengan menggunakan data yang terdapat pada datasets yanng sudah disediakan oleh orang lain")
    st.header("Data Sample")
    df = pd.read_csv("https://raw.githubusercontent.com/Wing-Dimas/datamining/main/seattle-weather.csv")
    st.text("""
    menggunakan kolom:
    * precipitation     : jumlah endapan awan
    * tempmax           : jumlah temperatur maksimal
    * tempmin           : jumlah temperatur minimal
    * wind              : kecepatan angin
    cuaca yang akan di prediksi:
    * drizzle (gerimis)
    * rain (hujan)
    * sun (panas)
    * snow (salju)
    * fog (kabut)
    """)
    st.caption('link datasets : https://www.kaggle.com/datasets/ananthr1/weather-prediction')
    st.dataframe(df)
    row, col = df.shape
    st.caption(f"({row} rows, {col} cols)")

    st.session_state["data"] = df