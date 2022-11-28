import streamlit as st
import plotly.express as px
 
# adding Folder_2/subfolder to the system path
from pages.home import *
from pages.preprocessing import *
from pages.modelling import *
from pages.implementasi import *

# intial template
px.defaults.template = "plotly_dark"
px.defaults.color_continuous_scale = "reds"

st.set_page_config(
    page_title="Weather Prediction",
    page_icon="⛅",
    layout="centered",
    initial_sidebar_state="expanded",
)

tab1, tab2, tab3, tab4 = st.tabs(["Home", "Preprocessing", "Modelling","Implementasi"])

with tab1:
    home()

with tab2:
    preprocessing()

with tab3:
    modelling()
    
with tab4:
    implementasi()