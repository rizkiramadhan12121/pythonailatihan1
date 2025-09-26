# Program Prediksi Tipe Kepribadian Sederhana dengan Scikit-learn
# Ini adalah contoh sederhana untuk demonstrasi, bukan model akurat.

# Langkah 1: Mengimpor library yang dibutuhkan
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Langkah 2: Menyiapkan Data Pelatihan
# Data input (fitur):
# Kolom: [suka_bersosialisasi (1-10), terorganisir (1-10), suka_berpikir_abstrak (1-10)]
# Label target: 0 = Introvert/Terorganisir (contoh), 1 = Ekstrovert/Spontan (contoh)
X = np.array([
    [9, 4, 3],  # Ekstrovert, Spontan, Praktis
    [2, 8, 7],  # Introvert, Terorganisir, Teoritis
    [8, 3, 2],  # Ekstrovert, Spontan, Praktis
    [3, 9, 8],  # Introvert, Terorganisir, Teoritis
    [10, 5, 4], # Ekstrovert, Spontan, Praktis
    [1, 7, 9],  # Introvert, Terorganisir, Teoritis
    [7, 6, 5],  # Ekstrovert, Spontan, Praktis
    [4, 10, 10],# Introvert, Terorganisir, Teoritis
])

# Target atau label untuk setiap data
# 0 = Introvert/Terorganisir, 1 = Ekstrovert/Spontan
y = np.array([1, 0, 1, 0, 1, 0, 1, 0])

# Langkah 3: Melatih Model AI
# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Membuat dan melatih model Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Langkah 4: Meminta input dari pengguna dan melakukan prediksi
print("--- Prediksi Tipe Kepribadian Sederhana (AI) ---")
print("Jawab pertanyaan berikut dengan skala 1-10 (1 = Sangat tidak setuju, 10 = Sangat setuju)")
try:
    sosialisasi = float(input("Saya suka bersosialisasi: "))
    organisir = float(input("Saya orang yang terorganisir: "))
    abstrak = float(input("Saya suka ide-ide abstrak: "))

    # Mengubah input menjadi format yang bisa diproses model
    data_pengguna = np.array([[sosialisasi, organisir, abstrak]])
    
    # Melakukan prediksi dengan model AI
    prediksi = model.predict(data_pengguna)
    
    # Menampilkan hasil prediksi
    print("\n--- Hasil Prediksi AI ---")
    if prediksi[0] == 0:
        print("Berdasarkan model AI, Anda diprediksi memiliki kecenderungan ke arah: Introvert dan Terorganisir.")
    else:
        print("Berdasarkan model AI, Anda diprediksi memiliki kecenderungan ke arah: Ekstrovert dan Spontan.")
        
except ValueError:
    print("\nInput tidak valid. Pastikan Anda memasukkan angka.")

print("\n----------------------------------")
print("Catatan: Ini adalah contoh sederhana. Hasil prediksi tidak akurat dan bukan diagnosis psikologis.")