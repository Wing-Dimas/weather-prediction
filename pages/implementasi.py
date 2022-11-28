import streamlit as st
import numpy as np
import pandas as pd
from sklearn.utils.validation import joblib

def implementasi():
    st.title("Implementasi")
    col1, col2 = st.columns(2)
    with col1:
        precipitation = st.number_input("precipitation (pengendapan)", 0.0, 60.0, step=0.1)
    with col2:
        temp_min = st.number_input("temperatur min", -5.0, 50.0, step=0.1)
        
    col3, col4 = st.columns(2)
    with col3:
        temp_max = st.number_input("temperatur max", -5.0, 50.0, step=0.1)
    with col4:
        wind = st.number_input("wind (kecepatan angin)", 0.0, 20.0, step=0.1)

    # section output
    st.subheader("Hasil Predict Cuaca")
    def submit():
        # input
        inputs = np.array([[precipitation, temp_max, temp_min, wind]])
        inp = pd.DataFrame(inputs, columns=["precipitation", "temp max", "temp min", "wind"])
        st.dataframe(inp)

        # import label encoder
        le = joblib.load("le.save")

        # create 3 output
        if "model" in  st.session_state:
            choose_model = st.session_state.model
            if choose_model == "Gaussian Naive Bayes":
                model = joblib.load("nb.joblib")
                st.subheader("Gaussian Naive Bayes")
            elif choose_model == "KNN":
                st.subheader("KNN")
                model = joblib.load("knn.joblib")
            else:
                st.subheader("Decision Tree")
                model = joblib.load("tree.joblib")
            y_pred = model.predict(inputs)
            st.success(f"Result : {le.inverse_transform(y_pred)[0]}", icon="✅")
        else:
            st.error("Silahkan pilih model terlebih dahulu di bagian tab modelling", icon="❌")

    # create button submit
    submitted = st.button("Submit")
    if submitted:
        submit()