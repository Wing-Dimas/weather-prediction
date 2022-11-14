import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
from sklearn.utils.validation import joblib

# intial template
px.defaults.template = "plotly_dark"
px.defaults.color_continuous_scale = "reds"

# create content
st.title("Predict Credit Score")
st.header("Data Sample")
st.sidebar.title("Input Data")

# read data
data = pd.read_csv("https://raw.githubusercontent.com/Wing-Dimas/datamining/main/credit_score.csv")
st.dataframe(data)

# create input
pendapatan_setahun = st.sidebar.text_input("Pendapatan Setahun(juta)")
kpr = st.sidebar.radio("KPR", ("aktif", "tidak aktif"))
jumlah_tanggungan = st.sidebar.text_input("Jumlah Tanggungan")
durasi_pinjaman = st.sidebar.selectbox("Durasi (bulan)", ("12", "36", "48"))
overdue = st.sidebar.selectbox("Overdue", ("0 - 30 days", "31 - 45 days", "46 - 60 days", "61 - 90 days", "> 90 days"))


# section output
st.subheader("Hasil Predict Score")
def submit():
    # cek input 
    scaler = joblib.load("scaler.save")
    normalize = scaler.transform([[int(pendapatan_setahun),int(durasi_pinjaman), int(jumlah_tanggungan)]])[0].tolist()

    kpr_ya = 0
    kpr_tidak = 0
    if kpr == "aktif":
        kpr_ya = 1
    else:
        kpr_tidak = 1


    overdues = [0,0,0,0,0]
    if(overdue == "0 - 30 days"):
        overdues[0] = 1
    elif(overdue == "31 - 45 days"):
        overdues[1] = 1
    elif(overdue == "46 - 60 days"):
        overdues[2] = 1
    elif(overdue == "61 - 90 days"):
        overdues[3] = 1
    else:
        overdues[4] = 1

    # create data input
    data_input = {
        "pendapatan_setahun_juta" : normalize[0],
        "durasi_pinjaman_bulan" : normalize[1],
        "jumlah_tanggungan" : normalize[2],
        "overdue_0 - 30 days": overdues[0],
        "overdue_31 - 45 days": overdues[1],
        "overdue_46 - 60 days": overdues[2],
        "overdue_61 - 90 days": overdues[3],
        "overdue_> 90 days": overdues[4],
        "KPR_TIDAK" : kpr_tidak,
        "KPR_YA": kpr_ya
    }

    inputs = np.array([[val for val in data_input.values()]])

    model = joblib.load("model.joblib")
    # with open("model.sav", "rb") as model_buffer:
        # model = pickle.load(model_buffer)
    pred = model.predict(inputs)
    st.text("Risk Rating : " + str(pred))


# create button submit
submitted = st.sidebar.button("Submit")
if submitted:
    submit()