import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📊 Visualisasi Data Churn")

# Load data
@st.cache_data
def tampilkan_visualisasi():
    return pd.read_csv("churn.csv")

df = tampilkan_visualisasi()
st.success("Data berhasil dimuat!")

# Preview Data
st.subheader("👀 Preview Data")
st.dataframe(df.head())

# Visualisasi Kolom Numerik
st.subheader("📈 Visualisasi Kolom Numerik")
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

if numeric_cols:
    selected_col = st.selectbox("Pilih kolom numerik", numeric_cols)

    fig, ax = plt.subplots()
    sns.histplot(df[selected_col], kde=True, ax=ax, color="skyblue")
    ax.set_title(f"Distribusi: {selected_col}")
    st.pyplot(fig)
else:
    st.warning("Tidak ditemukan kolom numerik.")

# Korelasi antar fitur
st.subheader("🔍 Korelasi Antar Fitur")
if len(numeric_cols) >= 2:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
else:
    st.warning("Tidak cukup fitur numerik untuk membuat heatmap.")
