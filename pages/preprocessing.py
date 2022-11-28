import streamlit as st
from sklearn.preprocessing import LabelEncoder

def preprocessing():
    st.subheader("Preprocessing Data")
    if "data" in st.session_state:
        df = st.session_state.data

        df = df.drop(columns="date")

        X = df.drop(columns="weather")
        y = df.weather

        le = LabelEncoder()
        le.fit(y)
        y = le.transform(y)
        y = le.inverse_transform(y)

        st.subheader("Output")
        col1, col2 = st.columns(2)

        with col1:
            st.write("X")
            st.dataframe(X)
        
        with col2:
            st.write("y")
            st.dataframe(y)

        st.session_state["X"] = X
        st.session_state["y"] = y