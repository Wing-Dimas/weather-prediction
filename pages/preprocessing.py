import streamlit as st
import pandas as pd
import numpy as np

def preprocessing():
    if "data" in st.session_state:
        df = st.session_state.data
        st.dataFrame(df)