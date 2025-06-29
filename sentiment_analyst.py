# -*- coding: utf-8 -*-
"""
Analisis Sentimen Ulasan Honkai Impact 3rd
Proyek Analisis Sentimen untuk mengklasifikasikan ulasan game Honkai Impact 3rd
dari Google Play Store menjadi sentimen positif, negatif, dan netral.

Dataset: Ulasan Honkai Impact 3rd dari Google Play Store
Metode: Machine Learning dan Deep Learning (LSTM)
"""

import pandas as pd
import numpy as np
import os

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

"""# Gathering Data"""

# Memuat dataset Honkai Impact 3rd yang sudah diproses
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, 'datasets', 'ulasan_honkai_impact_3_processed.csv')

print("Memuat dataset Honkai Impact 3rd...")
df = pd.read_csv(dataset_path)

print(f"Dataset berhasil dimuat!")
print(f"Jumlah data: {len(df)} sampel")
print(f"Kolom yang tersedia: {list(df.columns)}")

# Menampilkan informasi dasar dataset
print("\n=== INFORMASI DATASET ===")
print(f"Ukuran dataset: {df.shape}")
print(f"\nDistribusi kelas sentimen:")
print(df['polarity'].value_counts())

# Menghapus missing values dan duplikasi
print("\n=== PEMBERSIHAN DATA ===")
print(f"Data sebelum pembersihan: {len(df)} sampel")

df = df.dropna(subset=['text_final', 'polarity'])
df = df.drop_duplicates(subset=['text_final'])

print(f"Data setelah pembersihan: {len(df)} sampel")

# Pastikan kita memiliki minimal 3000 sampel sesuai kriteria
if len(df) >= 3000:
    print(f"✓ Dataset memenuhi kriteria minimal 3000 sampel ({len(df)} sampel)")
else:
    print(f"⚠ Dataset kurang dari 3000 sampel ({len(df)} sampel)")

"""# Data Splitting dan Ekstraksi Fitur"""

# Memisahkan fitur dan target
# Menggunakan 'text_final' yang sudah diproses sebagai fitur
X = df['text_final']  # Text yang sudah diproses
y = df['polarity']    # Label sentimen

print(f"\n=== PERSIAPAN DATA UNTUK MODELING ===")
print(f"Jumlah fitur teks: {len(X)}")
print(f"Distribusi label:")
print(y.value_counts())

"""### **Ekstraksi Fitur**

**TF-IDF**
"""

print("\n=== EKSTRAKSI FITUR TF-IDF ===")
# Inisialisasi TfidfVectorizer dengan 1000 unigram dan bigram terpenting yang muncul minimal di 5 dokumen namun tidak lebih dari 85% dokumen
tfidf_vectorizer = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.85, ngram_range=(1,2))
# Transformasi menjadi vektor TF-IDF
tfidf_matrix = tfidf_vectorizer.fit_transform(X)
# Konversi menjadi Dataframe
X_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

print(f"Dimensi fitur TF-IDF: {X_tfidf.shape}")

"""**Bag of Words (BoW)**"""

print("\n=== EKSTRAKSI FITUR BAG OF WORDS ===")
# Inisialisasi vectorizer dengan 1000 fitur terpenting yang muncul minimal di 5 dokumen namun tidak lebih dari 85% dokumen
vectorizer = CountVectorizer(max_features=1000, min_df=5, max_df=0.85)
# Konversi menjadi vektor BoW
bow_matrix = vectorizer.fit_transform(X)
# Konversi menjadi dataframe
X_bow = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())

print(f"Dimensi fitur BoW: {X_bow.shape}")

"""# Modeling

Sesuai dengan kriteria submission, akan dilakukan 3 percobaan skema pelatihan yang berbeda:
1. Logistic Regression dengan TF-IDF (80/20)
2. Random Forest dengan TF-IDF (70/30)  
3. Random Forest dengan BoW (80/20)
"""

print("\n=== PERSIAPAN MODELING ===")

# Membagi data menjadi data latih dan data uji dengan berbagai skema

print("Membagi data untuk berbagai skema pelatihan...")

# Skema 1: Logistic Regression | TF-IDF | 80/20
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Skema 2: Random Forest | TF-IDF | 70/30
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

# Skema 3: Random Forest | BoW | 80/20
X_train3, X_test3, y_train3, y_test3 = train_test_split(X_bow, y, test_size=0.2, random_state=42)

print("✓ Data berhasil dibagi untuk 3 skema pelatihan")

"""### **Percobaan 1: Logistic Regression dengan TF-IDF (80/20)**"""

print("\n=== PERCOBAAN 1: LOGISTIC REGRESSION + TF-IDF (80/20) ===")

# Melatih model Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train1, y_train1)

# Prediksi
y_train_pred1 = lr.predict(X_train1)
y_test_pred1 = lr.predict(X_test1)

# Evaluasi
train_acc1 = accuracy_score(y_train1, y_train_pred1)
test_acc1 = accuracy_score(y_test1, y_test_pred1)

result1 = {
    'Model': 'Logistic Regression (TF-IDF 80/20)',
    'Train Accuracy': train_acc1,
    'Test Accuracy': test_acc1
}

print(f"Train Accuracy: {train_acc1:.4f}")
print(f"Test Accuracy: {test_acc1:.4f}")

"""### **Percobaan 2: Random Forest dengan TF-IDF (70/30)**"""

print("\n=== PERCOBAAN 2: RANDOM FOREST + TF-IDF (70/30) ===")

# Melatih model Random Forest
rf_tfidf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_tfidf.fit(X_train2, y_train2)

# Prediksi
y_train_pred2 = rf_tfidf.predict(X_train2)
y_test_pred2 = rf_tfidf.predict(X_test2)

# Evaluasi
train_acc2 = accuracy_score(y_train2, y_train_pred2)
test_acc2 = accuracy_score(y_test2, y_test_pred2)

result2 = {
    'Model': 'Random Forest (TF-IDF 70/30)',
    'Train Accuracy': train_acc2,
    'Test Accuracy': test_acc2
}

print(f"Train Accuracy: {train_acc2:.4f}")
print(f"Test Accuracy: {test_acc2:.4f}")

"""### **Percobaan 3: Random Forest dengan BoW (80/20)**"""

print("\n=== PERCOBAAN 3: RANDOM FOREST + BOW (80/20) ===")

# Melatih model Random Forest dengan BoW
rf_bow = RandomForestClassifier(n_estimators=100, random_state=42)
rf_bow.fit(X_train3, y_train3)

# Prediksi
y_train_pred3 = rf_bow.predict(X_train3)
y_test_pred3 = rf_bow.predict(X_test3)

# Evaluasi
train_acc3 = accuracy_score(y_train3, y_train_pred3)
test_acc3 = accuracy_score(y_test3, y_test_pred3)

result3 = {
    'Model': 'Random Forest (BoW 80/20)',
    'Train Accuracy': train_acc3,
    'Test Accuracy': test_acc3
}

print(f"Train Accuracy: {train_acc3:.4f}")
print(f"Test Accuracy: {test_acc3:.4f}")

# Menampilkan ringkasan hasil 3 percobaan
print("\n=== RINGKASAN HASIL 3 PERCOBAAN ===")
results_ml = [result1, result2, result3]
accuracy_df_ml = pd.DataFrame(results_ml)
print(accuracy_df_ml.to_string(index=False))

"""### **Percobaan 4: Support Vector Machine (SVM)**"""

print("\n=== PERCOBAAN 4: SUPPORT VECTOR MACHINE (SVM) ===")

# Menggunakan SVM sebagai algoritma keempat
print("Melatih model SVM dengan TF-IDF...")

# Menggunakan data TF-IDF yang sama dengan skema 80/20
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train1, y_train1)

# Prediksi
y_train_pred_svm = svm_model.predict(X_train1)
y_test_pred_svm = svm_model.predict(X_test1)

# Evaluasi
train_acc_svm = accuracy_score(y_train1, y_train_pred_svm)
test_acc_svm = accuracy_score(y_test1, y_test_pred_svm)

print(f"Train Accuracy: {train_acc_svm:.4f}")
print(f"Test Accuracy: {test_acc_svm:.4f}")

# Menyimpan hasil evaluasi
result4 = {
    'Model': 'Support Vector Machine (TF-IDF 80/20)',
    'Train Accuracy': train_acc_svm,
    'Test Accuracy': test_acc_svm
}

print("\n=== HASIL PERCOBAAN SVM ===")
accuracy_df_svm = pd.DataFrame([result4])
print(accuracy_df_svm.to_string(index=False))

"""# **Inference dan Evaluasi Akhir**"""

print("\n=== RINGKASAN SEMUA PERCOBAAN ===")

# Menggabungkan semua hasil
final_results = [result1, result2, result3, result4]
accuracy_df_final = pd.DataFrame(final_results).sort_values(by='Test Accuracy', ascending=False)

print(accuracy_df_final.to_string(index=False))

# Cek apakah memenuhi kriteria minimum 85%
print(f"\n=== EVALUASI KRITERIA SUBMISSION ===")
models_above_85 = accuracy_df_final[accuracy_df_final['Test Accuracy'] >= 0.85]
print(f"Jumlah model dengan akurasi ≥ 85%: {len(models_above_85)} dari 4 model")

if len(models_above_85) >= 3:
    print("✓ Memenuhi kriteria: minimal 3 model dengan akurasi testing ≥ 85%")
else:
    print("⚠ Belum memenuhi kriteria: perlu minimal 3 model dengan akurasi testing ≥ 85%")

# Model terbaik
best_model_info = accuracy_df_final.iloc[0]
print(f"\nModel terbaik: {best_model_info['Model']}")
print(f"Test Accuracy: {best_model_info['Test Accuracy']:.4f}")

"""### **Fungsi Prediksi untuk Inference**"""

def predict_sentiment_lstm(text, model, tokenizer, label_encoder, max_length=100):
    """
    Fungsi untuk memprediksi sentimen dari teks input menggunakan model LSTM
    
    Args:
        text (str): Teks yang akan diprediksi
        model: Model LSTM yang sudah dilatih
        tokenizer: Tokenizer yang sudah difit
        label_encoder: LabelEncoder yang sudah difit
        max_length (int): Panjang maksimum sequence
    
    Returns:
        tuple: (predicted_class, confidence_scores)
    """
    # Preprocessing teks input
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_length, padding='post')
    
    # Prediksi
    probs = model.predict(padded, verbose=0)
    pred_class_idx = np.argmax(probs, axis=1)[0]
    
    # Konversi kembali ke label asli
    pred_class = label_encoder.inverse_transform([pred_class_idx])[0]
    confidence = probs[0][pred_class_idx]
    
    return pred_class, confidence, probs[0]

def predict_sentiment_ml(text, model, vectorizer, model_type='tfidf'):
    """
    Fungsi untuk memprediksi sentimen menggunakan model machine learning tradisional
    
    Args:
        text (str): Teks yang akan diprediksi
        model: Model ML yang sudah dilatih
        vectorizer: Vectorizer (TF-IDF atau BoW) yang sudah difit
        model_type (str): Jenis vectorizer ('tfidf' atau 'bow')
    
    Returns:
        tuple: (predicted_class, confidence_score)
    """
    # Transform teks menggunakan vectorizer
    text_vector = vectorizer.transform([text])
    
    # Prediksi
    pred_class = model.predict(text_vector)[0]
    
    # Mendapatkan confidence score (probability)
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(text_vector)[0]
        confidence = max(probs)
    else:
        confidence = None
    
    return pred_class, confidence

"""### **Contoh Testing dan Demonstrasi Inference**"""

print("\n=== DEMONSTRASI INFERENCE ===")

# Contoh teks untuk testing
test_texts = [
    "Game ini sangat bagus dan menyenangkan untuk dimainkan!",
    "Aplikasi sering crash dan sangat mengecewakan",
    "Grafik bagus tapi gameplaynya biasa saja",
    "Honkai Impact 3rd adalah game terbaik yang pernah saya mainkan",
    "Bug nya banyak sekali, tidak recommended"
]

print("Menguji berbagai model dengan contoh teks...")

for i, text in enumerate(test_texts, 1):
    print(f"\n--- Contoh {i} ---")
    print(f"Teks: '{text}'")
    
    # Prediksi dengan LSTM (model terbaik)
    pred_lstm, conf_lstm, probs_lstm = predict_sentiment_lstm(
        text, model, tokenizer, label_encoder, max_length
    )
    print(f"LSTM: {pred_lstm} (confidence: {conf_lstm:.3f})")
    
    # Prediksi dengan Logistic Regression + TF-IDF
    pred_lr, conf_lr = predict_sentiment_ml(text, lr, tfidf_vectorizer, 'tfidf')
    print(f"Logistic Regression: {pred_lr} (confidence: {conf_lr:.3f})")
    
    # Prediksi dengan Random Forest + BoW
    pred_rf, conf_rf = predict_sentiment_ml(text, rf_bow, vectorizer, 'bow')
    print(f"Random Forest: {pred_rf} (confidence: {conf_rf:.3f})")

"""### **Kesimpulan**"""

print(f"\n=== KESIMPULAN ANALISIS SENTIMEN HONKAI IMPACT 3RD ===")
print(f"✓ Dataset: {len(df)} ulasan dari Google Play Store")
print(f"✓ Fitur yang digunakan: Text yang sudah diproses (preprocessing)")
print(f"✓ 3 kelas sentimen: {', '.join(label_encoder.classes_)}")
print(f"✓ 4 percobaan model dengan kombinasi algoritma dan fitur berbeda")
print(f"✓ Model terbaik: {best_model_info['Model']} dengan akurasi {best_model_info['Test Accuracy']:.4f}")

if best_model_info['Test Accuracy'] >= 0.92:
    print(f"✓ Mencapai target akurasi tinggi (>92%)")
elif best_model_info['Test Accuracy'] >= 0.85:
    print(f"✓ Memenuhi kriteria minimum akurasi (≥85%)")
else:
    print(f"⚠ Belum mencapai kriteria minimum akurasi (85%)")

print(f"\nModel ini dapat digunakan untuk menganalisis sentimen ulasan game")
print(f"dan memberikan insight tentang persepsi pengguna terhadap Honkai Impact 3rd.")