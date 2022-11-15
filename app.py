import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
from sklearn.utils.validation import joblib

# intial template
px.defaults.template = "plotly_dark"
px.defaults.color_continuous_scale = "reds"

# create content
st.title("Weather Prediction")
st.container()
st.write("Website ini bertujuan untuk memprediksi cuaca yang akan terjadi, dengan menggunakan data yang terdapat pada datasets yanng sudah disediakan oleh orang lain")
st.header("Data Sample")
st.sidebar.title("Input Data")

# read data
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

# create input
precipitation = st.sidebar.number_input("precipitation (pengendapan)", 0.0, 60.0, step=0.1)
temp_min = st.sidebar.number_input("temperatur min", -5.0, 50.0, step=0.1)
temp_max = st.sidebar.number_input("temperatur max", -5.0, 50.0, step=0.1)
wind = st.sidebar.number_input("wind (kecepatan angin)", 0.0, 20.0, step=0.1)


# section output
st.subheader("Hasil Predict Cuaca")
def submit():
    # input
    inputs = np.array([[precipitation, temp_max, temp_min, wind]])
    st.write(inputs)

    # import label encoder
    le = joblib.load("le.save")

    # create 3 output
    col1, col2, col3 = st.columns(3)
    with col1:
        model1 = joblib.load("nb.joblib")
        y_pred1 = model1.predict(inputs)
        col1.subheader("Gaussian Naive Bayes")
        col1.write(f"Result : {le.inverse_transform(y_pred1)[0]}")


    with col2:
        model2 = joblib.load("knn.joblib")
        y_pred2 = model2.predict(inputs)
        st.subheader("k-nearest neighbors")
        col2.write(f"Result : {le.inverse_transform(y_pred2)[0]}")

    with col3:
        model3 = joblib.load("tree.joblib")
        y_pred3 = model3.predict(inputs)
        st.subheader("Decision Tree")
        col3.write(f"Result : {le.inverse_transform(y_pred3)[0]}")


# create button submit
submitted = st.sidebar.button("Submit")
if submitted:
    submit()