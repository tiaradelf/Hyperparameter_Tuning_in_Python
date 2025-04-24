import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title("🔮 Prediksi Customer Churn (ML)")

# Load data
@st.cache_data
def prediksi():
    return pd.read_csv("churn.csv")

df = prediksi()
st.dataframe(df)

# Preprocessing sederhana (bisa disesuaikan)
# Drop kolom yang tidak berguna jika ada
X = df.select_dtypes(include=["int64", "float64"]).drop(columns=["Churn"], errors="ignore")
if "Churn" in df.columns:
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})  # ubah ke angka
    y = df["Churn"]
else:
    y = None

# Training model sederhana (real-case: load dari joblib)
@st.cache_resource
def train_model(X, y):
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

if y is not None:
    model = train_model(X, y)

    st.subheader("🧾 Masukkan Data untuk Prediksi")

    user_input = {}
    for col in X.columns:
        user_input[col] = st.number_input(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))

    input_df = pd.DataFrame([user_input])

    # Prediksi
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][prediction]

    st.subheader("📢 Hasil Prediksi")
    if prediction == 1:
        st.error(f"⚠️ Pelanggan diprediksi akan **CHURN** ({prediction_proba:.2%} confidence)")
    else:
        st.success(f"✅ Pelanggan **TIDAK akan churn** ({prediction_proba:.2%} confidence)")

else:
    st.warning("Kolom target 'Churn' tidak ditemukan di dataset.")
