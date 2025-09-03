import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Who is the Best F1 Driver?",
    page_icon="🏎️",
    layout="wide"
)

st.title("🏁 Who is the Best F1 Driver?")
st.write("A data-driven dissertation project using Bayesian analysis and the Driver Performance Index (DPI).")

@st.cache_data
def load_data():
    data = pd.DataFrame({
        "Driver": ["Fangio", "Schumacher", "Hamilton", "Verstappen"],
        "DPI": [92.3, 91.7, 90.9, 89.4]
    })
    return data

df = load_data()
st.subheader("Driver Performance Index (sample data)")
st.dataframe(df, use_container_width=True)
st.bar_chart(df.set_index("Driver")["DPI"])
