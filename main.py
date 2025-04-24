import streamlit as st

st.set_page_config(page_title="Portfolio", layout="wide", page_icon=":rocket:")
st.title("My Portfolio")
st.header("Hyperparameter Tuning in Python")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["About Me", "Project", "Machine Learning", "Contact"])

if page == "About Me":
    import About_Me
    About_Me.tampilkan_about_me()
elif page == "Contact":
    import Contact
    Contact.tampilkan_Contact()
elif page == "Project":
    import visualisasi
    visualisasi.tampilkan_visualisasi()
elif page == "Machine Learning":
    import prediksi
    prediksi.prediksi()