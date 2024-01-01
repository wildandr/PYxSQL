import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.svm import SVC
from sqlalchemy import create_engine

# Fungsi untuk mendapatkan rekomendasi obat
def get_rekomendasi(input_condition, input_age, rekom, tfidf, svm, tfidf_rekom):
    # Membangun vektor TF-IDF dari input_condition
    input_condition_vec = tfidf.transform([input_condition])

    # Melakukan prediksi menggunakan model SVM
    # Pastikan dimensi vektor support cocok dengan dimensi input_condition_vec
    input_data_tfidf = input_condition_vec[:, svm.support_[:input_condition_vec.shape[1]]]
    predicted_rekom_id = svm.predict(input_data_tfidf)[0]

    # Memfilter rekomendasi obat
    rekomendasi_svm = rekom[rekom['Kd_id'] == predicted_rekom_id]

    # Menggunakan collaborative filtering untuk mendapatkan rekomendasi obat lainnya:
    # Menghitung similarity score antara obat yang direkomendasikan oleh SVM dan semua obat di database
    rekom_matrix = tfidf_rekom.transform(rekom['Brand'])
    input_data_matrix = tfidf_rekom.transform([input_condition])

    cosine_similarities = linear_kernel(input_data_matrix, rekom_matrix)

    # Mengambil 10 obat dengan similarity score tertinggi
    top_similarities = np.argsort(cosine_similarities.flatten())[-10:][::-1]
    rekomendasi_collaborative_filtering = rekom.iloc[top_similarities]

    return rekomendasi_svm, rekomendasi_collaborative_filtering

# Mengatur koneksi ke database
db_url = "mysql+mysqlconnector://root:Makanmakan3x*@localhost:3306/obdb"
engine = create_engine(db_url)

# Membaca tabel input
input_df = pd.read_sql_query("SELECT * FROM input", con=engine)

# Membaca tabel kd_age
kd_age = pd.read_sql_query("SELECT * FROM kd_age", con=engine)

# Membaca tabel kd_cond
kd_cond = pd.read_sql_query("SELECT * FROM kd_cond", con=engine)

# Membaca tabel kd_dosage
kd_dosage = pd.read_sql_query("SELECT * FROM kd_dosage", con=engine)

# Membaca tabel rekom
rekom = pd.read_sql_query("SELECT * FROM rekom", con=engine)

# Pastikan "Kd_Cond" dalam input_df memiliki tipe data yang sesuai
input_df['Kd_Cond'] = input_df['Kd_Cond'].astype(str)

# Mengaplikasikan operasi split pada string
input_df['Kd_Cond'] = input_df['Kd_Cond'].str.split(',').apply(lambda x: [int(i) for i in x][0] if x[0] else None)
input_df['Kd_Cond'] = input_df['Kd_Cond'].astype(float).astype(pd.Int64Dtype())
input_df['Kd_Cond'] = pd.to_numeric(input_df['Kd_Cond'], errors='coerce', downcast='integer')

# Melakukan join antara input_df dengan tabel kd_age, kd_cond, dan kd_dosage untuk mendapatkan informasi obat:
input_df = input_df.merge(kd_age, on='Kd_Age')
input_df = input_df.merge(kd_cond, on='Kd_Cond', how='inner')

# Membangun vektor TF-IDF dari kolom 'Condition in code'
tfidf = TfidfVectorizer()
condition_matrix = tfidf.fit_transform(rekom['Kd_Cond'])
print(condition_matrix)

# Melakukan prediksi menggunakan model SVM
svm = SVC(kernel='linear')
svm.fit(condition_matrix, rekom['Kd_id'])

# Membangun vektor TF-IDF dari kolom 'Brand' pada data obat rekomendasi
tfidf_rekom = TfidfVectorizer()
rekom_matrix = tfidf_rekom.fit_transform(rekom['Brand'])
print(rekom_matrix)

# Memanggil fungsi untuk mendapatkan rekomendasi
input_condition = "2, 5, 6, 10, 11"  # Ganti dengan input condition in code sesuai dengan input user
input_age = 15  # Ganti dengan input umur anak sesuai dengan input user
rekomendasi_svm, rekomendasi_collaborative_filtering = get_rekomendasi(input_condition, input_age, rekom, tfidf, svm, tfidf_rekom)

# Menampilkan hasil rekomendasi
print("Rekomendasi SVM:")
print(rekomendasi_svm[['Brand', 'Manufacture', 'Dosage in code', 'Ingredients', 'Caution', 'Side Effect', 'Contra Indication']])

print("\nRekomendasi Collaborative Filtering:")
print(rekomendasi_collaborative_filtering[['Brand', 'Manufacture', 'Dosage in code', 'Ingredients', 'Caution', 'Side Effect', 'Contra Indication']])

# Memanggil fungsi untuk mendapatkan rekomendasi
input_condition = "4, 5, 6, 7, 2, 8, 9, 10, 11, 12, 13"  # Ganti dengan input condition in code sesuai dengan input user
input_age = 10  # Ganti dengan input umur anak sesuai dengan input user
rekomendasi_svm, rekomendasi_collaborative_filtering = get_rekomendasi(input_condition, input_age, rekom, tfidf, svm, tfidf_rekom)

# Memberikan output yang acak menggunakan sample
jumlah_sampel_acak = 10  # Ganti dengan jumlah sampel yang diinginkan
rekomendasi_svm_acak = rekomendasi_svm.sample(n=min(jumlah_sampel_acak, len(rekomendasi_svm)), replace=True)
rekomendasi_collaborative_filtering_acak = rekomendasi_collaborative_filtering.sample(n=min(jumlah_sampel_acak, len(rekomendasi_collaborative_filtering)), replace=True)

# Cetak DataFrame yang acak
print("Rekomendasi SVM (Acak):")
print(rekomendasi_svm_acak[['Brand', 'Manufacture', 'Dosage in code', 'Ingredients', 'Caution', 'Side Effect', 'Contra Indication']])

print("\nRekomendasi Collaborative Filtering (Acak):")
print(rekomendasi_collaborative_filtering_acak[['Brand', 'Manufacture', 'Dosage in code', 'Ingredients', 'Caution', 'Side Effect', 'Contra Indication']])