import streamlit as st
import pandas as pd
import joblib

# --- Load model ---
with open("model_graduation.pkl", "rb") as file:
    model = joblib.load(file)

# --- Judul aplikasi ---
st.title("Prediksi Kategori Waktu Lulus Mahasiswa")

st.write("""
Aplikasi ini memprediksi apakah seorang mahasiswa akan lulus **Tepat Waktu** atau **Terlambat** berdasarkan beberapa variabel.
Silakan masukkan data di bawah ini:
""")

# --- Form input ---
with st.form("prediction_form"):
    new_ACT = st.number_input("Masukkan nilai ACT composite score:", min_value=0.0, step=0.1)
    new_SAT = st.number_input("Masukkan nilai SAT total score:", min_value=0.0, step=1.0)
    new_GPA = st.number_input("Masukkan nilai rata-rata SMA:", min_value=0.0, max_value=4.0, step=0.01)
    new_income = st.number_input("Masukkan pendapatan orang tua:", min_value=0.0, step=100.0)
    new_education = st.number_input("Masukkan tingkat pendidikan orang tua (angka):", min_value=0.0, step=1.0)

    submit = st.form_submit_button("Prediksi")

# --- Prediksi jika tombol ditekan ---
if submit:
    try:
        new_data_df = pd.DataFrame(
            [[new_ACT, new_SAT, new_GPA, new_income, new_education]],
            columns=['ACT composite score', 'SAT total score', 'high school gpa', 'parental income', 'parent_edu_numerical']
        )

        predicted_code = model.predict(new_data_df)[0]
        label_mapping = {1: 'On Time', 0: 'Late'}
        predicted_label = label_mapping.get(predicted_code, 'Tidak diketahui')

        st.success(f"**Prediksi kategori masa studi:** {predicted_label}")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")


