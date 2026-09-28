# Import library utama
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import fungsi dari scikit-learn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 1. Load dataset
df = pd.read_csv('dataset_tugas1_preprocessing.csv')

# 2. Cek struktur dan tipe data awal
print("--- Informasi Dataset ---")
print(df.info())

# 3. Tampilkan 5 baris pertama
print("\n--- 5 Baris Pertama ---")
print(df.head())

# 4. Cek jumlah nilai hilang (missing values) per kolom
print("\n--- Jumlah Missing Values per Kolom ---")
print(df.isnull().sum())

# 5. Statistik deskriptif
print("\n--- Statistik Deskriptif ---")
print(df.describe(include='all'))

# 1. Imputasi kolom Nilai_Akhir menggunakan MODUS (kategori paling sering muncul)
modus_nilai = df['Nilai_Akhir'].mode()[0]
df['Nilai_Akhir'] = df['Nilai_Akhir'].fillna(modus_nilai)

# 2. Imputasi kolom Umur menggunakan MEDIAN (nilai tengah)
median_umur = df['Umur'].median()
df['Umur'] = df['Umur'].fillna(median_umur)

# Verifikasi bahwa tidak ada lagi nilai kosong
print("Jumlah missing values setelah imputasi:")
print(df.isnull().sum())

# Konversi ke format datetime
df['Tanggal_Ujian'] = pd.to_datetime(df['Tanggal_Ujian'], format='mixed')

# Cek tipe data kolom Tanggal_Ujian
print("Tipe data Tanggal_Ujian:", df['Tanggal_Ujian'].dtype)

# Inisialisasi LabelEncoder
le = LabelEncoder()

# Buat copy dataframe untuk versi encoded
df_encoded = df.copy()

# Daftar kolom yang akan di-encode
categorical_cols = ['Nama', 'Jenis_Kelamin', 'Prodi', 'Status', 'Nilai_Akhir']

# Lakukan loop untuk encoding tiap kolom
for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])

# Tampilkan hasil setelah encoding
print("--- Dataframe Setelah Label Encoding ---")
print(df_encoded.head())

# Split data 80% train dan 20% test
train_df, test_df = train_test_split(df_encoded, test_size=0.2, random_state=42)

print(f"Jumlah Data Latih (Train Set) : {train_df.shape[0]} baris")
print(f"Jumlah Data Uji (Test Set)    : {test_df.shape[0]} baris")

# Mengatur ukuran figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1. Histogram Distribusi Umur
sns.histplot(df['Umur'], kde=True, ax=axes[0], color='skyblue', bins=10)
axes[0].set_title('Histogram Distribusi Umur Mahasiswa')
axes[0].set_xlabel('Umur')
axes[0].set_ylabel('Frekuensi')

# 2. Grafik Batang Jumlah Mahasiswa per Prodi
prodi_counts = df['Prodi'].value_counts()
sns.barplot(x=prodi_counts.index, y=prodi_counts.values, ax=axes[1], palette='viridis')
axes[1].set_title('Jumlah Mahasiswa per Program Studi')
axes[1].set_xlabel('Program Studi')
axes[1].set_ylabel('Jumlah Mahasiswa')
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.show()