# =============================================================================
# PROYEK SENTIMENT ANALYSIS - HONKAI IMPACT 3 GOOGLE PLAY REVIEWS
# =============================================================================
# 
# **Nama:** Faris Munir Mahdi
# **Dataset:** Google Play Store Reviews - Honkai Impact 3 Application
# **Domain Proyek:** Natural Language Processing and Sentiment Analysis
# **Target:** Minimal 3 model dengan akurasi testing ≥85%

# =============================================================================
# 1. IMPORT LIBRARIES DAN SETUP
# =============================================================================

# 1.1. Import library machine learning
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1.2. Import library TensorFlow untuk LSTM
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# 1.3. Konfigurasi GPU TensorFlow
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# 1.4. Aktifkan GPU memory growth untuk menghindari error OOM
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"⚠ GPU configuration error: {e}")
else:
    print("⚠ No GPU detected, using CPU")

# =============================================================================
# 2. LOAD DATASET DAN SETUP PATH
# =============================================================================

# 2.1. Setup path dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, 'datasets', 'ulasan_honkai_impact_3_processed.csv')

# 2.2. Load dataset yang sudah diproses
df = pd.read_csv(dataset_path)
print(f"✅ Dataset loaded: {len(df)} samples")

# =============================================================================
# 3. ANALISIS EKSPLORATORI DATA (EDA)
# =============================================================================

# 3.1. Informasi dasar dataset
print(f"\n--- Dataset Info ---")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# 3.2. Distribusi sentimen
print(f"\n--- Sentiment Distribution ---")
print(df['polarity'].value_counts())

# 3.3. Analisis missing values
print(f"\n--- Missing Values ---")
print(df.isnull().sum())

# =============================================================================
# 4. DATA CLEANING DAN PREPROCESSING
# =============================================================================

# 4.1. Pembersihan data
print(f"\n--- Data Cleaning ---")
print(f"Before cleaning: {len(df)} samples")

df = df.dropna(subset=['text_final', 'polarity'])
df = df.drop_duplicates(subset=['text_final'])

print(f"After cleaning: {len(df)} samples")

# 4.2. Validasi kriteria dataset
if len(df) >= 3000:
    print(f"✅ Dataset meets criteria (≥3000 samples)")
else:
    print(f"⚠ Dataset below 3000 samples")

# =============================================================================
# 5. PERSIAPAN FEATURE DAN TARGET
# =============================================================================

# 5.1. Definisi feature dan target
X = df['text_final']
y = df['polarity']

# 5.2. Informasi feature dan target
print(f"\n--- Feature & Target Info ---")
print(f"Text features: {len(X)}")
print(f"Target distribution:")
print(y.value_counts())

# =============================================================================
# 6. EKSTRAKSI FEATURE: TF-IDF VECTORIZATION
# =============================================================================

# 6.1. Setup TF-IDF Vectorizer
print(f"\n--- TF-IDF Vectorization ---")
tfidf_vectorizer = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.85, ngram_range=(1,2))
tfidf_matrix = tfidf_vectorizer.fit_transform(X)
X_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
print(f"✅ TF-IDF shape: {X_tfidf.shape}")

# =============================================================================  
# 7. EKSTRAKSI FEATURE: BAG OF WORDS VECTORIZATION
# =============================================================================

# 7.1. Setup Bag of Words Vectorizer
print(f"\n--- Bag of Words Vectorization ---")
vectorizer = CountVectorizer(max_features=1000, min_df=5, max_df=0.85)
bow_matrix = vectorizer.fit_transform(X)
X_bow = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())
print(f"✅ BoW shape: {X_bow.shape}")

# =============================================================================
# 8. PEMBAGIAN DATA UNTUK EKSPERIMEN
# =============================================================================

# 8.1. Setup pembagian data untuk 4 eksperimen
print(f"\n--- Data Splitting ---")
# Experiment 1: Logistic Regression | TF-IDF | 80/20
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Experiment 2: Random Forest | TF-IDF | 70/30
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

# Experiment 3: Random Forest | BoW | 80/20
X_train3, X_test3, y_train3, y_test3 = train_test_split(X_bow, y, test_size=0.2, random_state=42)

# 8.2. Inisialisasi container hasil
all_results = []
print("✅ Data split complete for 4 experiments")

# =============================================================================
# 9. PEMODELAN: LOGISTIC REGRESSION (TF-IDF)
# =============================================================================

# 9.1. Training model Logistic Regression
print(f"\n--- Experiment 1: Logistic Regression + TF-IDF ---")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train1, y_train1)

# 9.2. Prediksi dan evaluasi
y_train_pred1 = lr.predict(X_train1)
y_test_pred1 = lr.predict(X_test1)

train_acc1 = accuracy_score(y_train1, y_train_pred1)
test_acc1 = accuracy_score(y_test1, y_test_pred1)

# 9.3. Simpan hasil eksperimen 1
result1 = {'Model': 'Logistic Regression (TF-IDF)', 'Train Acc': train_acc1, 'Test Acc': test_acc1}
all_results.append(result1)
print(f"✅ Train: {train_acc1:.4f}, Test: {test_acc1:.4f}")

# =============================================================================
# 10. PEMODELAN: RANDOM FOREST (TF-IDF)
# =============================================================================

# 10.1. Training model Random Forest dengan TF-IDF
print(f"\n--- Experiment 2: Random Forest + TF-IDF ---")
rf_tfidf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_tfidf.fit(X_train2, y_train2)

# 10.2. Prediksi dan evaluasi
y_train_pred2 = rf_tfidf.predict(X_train2)
y_test_pred2 = rf_tfidf.predict(X_test2)

train_acc2 = accuracy_score(y_train2, y_train_pred2)
test_acc2 = accuracy_score(y_test2, y_test_pred2)

# 10.3. Simpan hasil eksperimen 2
result2 = {'Model': 'Random Forest (TF-IDF)', 'Train Acc': train_acc2, 'Test Acc': test_acc2}
all_results.append(result2)
print(f"✅ Train: {train_acc2:.4f}, Test: {test_acc2:.4f}")

# =============================================================================
# 11. PEMODELAN: RANDOM FOREST (BAG OF WORDS)
# =============================================================================

# 11.1. Training model Random Forest dengan BoW
print(f"\n--- Experiment 3: Random Forest + BoW ---")
rf_bow = RandomForestClassifier(n_estimators=100, random_state=42)
rf_bow.fit(X_train3, y_train3)

# 11.2. Prediksi dan evaluasi
y_train_pred3 = rf_bow.predict(X_train3)
y_test_pred3 = rf_bow.predict(X_test3)

train_acc3 = accuracy_score(y_train3, y_train_pred3)
test_acc3 = accuracy_score(y_test3, y_test_pred3)

# 11.3. Simpan hasil eksperimen 3
result3 = {'Model': 'Random Forest (BoW)', 'Train Acc': train_acc3, 'Test Acc': test_acc3}
all_results.append(result3)
print(f"✅ Train: {train_acc3:.4f}, Test: {test_acc3:.4f}")

# =============================================================================
# 12. PEMODELAN: LSTM (DEEP LEARNING)
# =============================================================================

# 12.1. Persiapan data untuk LSTM
print(f"\n--- Experiment 4: LSTM Deep Learning ---")

# Prepare data for LSTM
X_text = df['text_final']
y_text = df['polarity']

# 12.2. Encode labels ke nilai numerik
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_text)
num_classes = len(label_encoder.classes_)

# 12.3. Tokenisasi dengan 5000 kata paling sering
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(X_text)

# 12.4. Konversi teks ke sequences
sequences = tokenizer.texts_to_sequences(X_text)

# 12.5. Pad sequences ke panjang uniform 20
X_padded = pad_sequences(sequences, maxlen=20, padding='post')

# 12.6. One-hot encode labels untuk categorical crossentropy
y_categorical = to_categorical(y_encoded, num_classes=num_classes)

# 12.7. Split data untuk LSTM (80/20)
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
    X_padded, y_categorical, test_size=0.2, random_state=42
)

# 12.8. Build arsitektur LSTM
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    lstm_model = Sequential()
    lstm_model.add(Embedding(input_dim=5000, output_dim=64))  # Embedding layer
    lstm_model.add(LSTM(64, return_sequences=False))         # LSTM layer
    lstm_model.add(Dropout(0.5))                             # Dropout for regularization
    lstm_model.add(Dense(num_classes, activation='softmax')) # Output layer

# 12.9. Compile model
lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(f"✅ Model built on: {'GPU' if gpus else 'CPU'}")
print("LSTM Model Summary:")
lstm_model.summary()

# 12.10. Training model
print("⏳ Training LSTM model...")
print(f"Training on: {'GPU' if gpus else 'CPU'}")

# Use GPU for training if available
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    history = lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=64, 
                            validation_split=0.1, verbose=1)

# 12.11. Make predictions
y_train_pred_lstm = lstm_model.predict(X_train_lstm)
y_test_pred_lstm = lstm_model.predict(X_test_lstm)

# 12.12. Convert predictions back to class labels
y_train_pred_labels = np.argmax(y_train_pred_lstm, axis=1)
y_test_pred_labels = np.argmax(y_test_pred_lstm, axis=1)
y_train_true_labels = np.argmax(y_train_lstm, axis=1)
y_test_true_labels = np.argmax(y_test_lstm, axis=1)

# 12.13. Calculate accuracies
train_acc_lstm = accuracy_score(y_train_true_labels, y_train_pred_labels)
test_acc_lstm = accuracy_score(y_test_true_labels, y_test_pred_labels)

# 12.14. Simpan hasil eksperimen 4
result4 = {'Model': 'LSTM (Deep Learning)', 'Train Acc': train_acc_lstm, 'Test Acc': test_acc_lstm}
all_results.append(result4)
print(f"✅ Train: {train_acc_lstm:.4f}, Test: {test_acc_lstm:.4f}")

# =============================================================================
# 13. EVALUASI MODEL DAN PERBANDINGAN
# =============================================================================

# 13.1. Ringkasan performa semua model
print(f"\n--- Model Performance Summary ---")
accuracy_df_final = pd.DataFrame(all_results).sort_values(by='Test Acc', ascending=False)
print(accuracy_df_final.to_string(index=False))

# =============================================================================
# 14. VALIDASI KRITERIA SUBMISSION
# =============================================================================

# 14.1. Pengecekan kriteria submission
print(f"\n--- Submission Criteria Check ---")
models_above_85 = accuracy_df_final[accuracy_df_final['Test Acc'] >= 0.85]
print(f"Models with ≥85% accuracy: {len(models_above_85)}/4")

# 14.2. Evaluasi kriteria utama
if len(models_above_85) >= 3:
    print("✅ PASSED: ≥3 models with 85%+ accuracy")
else:
    print("⚠ FAILED: Need ≥3 models with 85%+ accuracy")

# 14.3. Informasi model terbaik
best_model_info = accuracy_df_final.iloc[0]
print(f"🏆 Best model: {best_model_info['Model']} ({best_model_info['Test Acc']:.4f})")

# 14.4. Pengecekan kriteria bonus
models_above_92 = accuracy_df_final[accuracy_df_final['Test Acc'] >= 0.92]
if len(models_above_92) >= 1:
    print("🎉 BONUS: ≥1 model with 92%+ accuracy")

# =============================================================================
# 15. FUNGSI INFERENCE UNTUK PREDIKSI
# =============================================================================

# 15.1. Fungsi prediksi untuk model tradisional
def predict_sentiment(text, model, vectorizer):
    """Prediksi sentimen menggunakan model tradisional (LR, RF)"""
    text_vector = vectorizer.transform([text])
    if hasattr(vectorizer, 'get_feature_names_out'):
        feature_names = vectorizer.get_feature_names_out()
        text_df = pd.DataFrame(text_vector.toarray(), columns=feature_names)
        pred_class = model.predict(text_df)[0]
        
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(text_df)[0]
            confidence = max(probs)
        else:
            confidence = 1.0
    else:
        text_array = text_vector.toarray()
        pred_class = model.predict(text_array)[0]
        
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(text_array)[0]
            confidence = max(probs)
        else:
            confidence = 1.0
    
    return pred_class, confidence

# 15.2. Fungsi prediksi LSTM
def predict_lstm(text):
    """Prediksi sentimen menggunakan LSTM model"""
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=20, padding='post')
    with tf.device('/GPU:0' if gpus else '/CPU:0'):
        probs = lstm_model.predict(padded, verbose=0)  # Add verbose=0 to reduce output
    pred_class = np.argmax(probs, axis=1)[0]
    confidence = max(probs[0])
    return label_encoder.inverse_transform([pred_class])[0], confidence

# =============================================================================
# 16. TESTING INFERENCE PADA CONTOH TEKS
# =============================================================================

# 16.1. Contoh teks untuk testing
print(f"\n--- Inference Testing ---")

test_texts = [
    "good game, fun mechanics",
    "Membosankan",
    "garbage company",
    "Honkai Impact 3rd game terbaik!",
    "bring back part",
]

# 16.2. Testing semua model pada contoh teks
for i, text in enumerate(test_texts, 1):
    print(f"\nTest {i}: '{text}'")
    
    pred_lr, conf_lr = predict_sentiment(text, lr, tfidf_vectorizer)
    print(f"Logistic Regression: {pred_lr} ({conf_lr:.3f})")
    
    pred_rf_tfidf, conf_rf_tfidf = predict_sentiment(text, rf_tfidf, tfidf_vectorizer)
    print(f"Random Forest (TF-IDF): {pred_rf_tfidf} ({conf_rf_tfidf:.3f})")
    
    pred_rf_bow, conf_rf_bow = predict_sentiment(text, rf_bow, vectorizer)
    print(f"Random Forest (BoW): {pred_rf_bow} ({conf_rf_bow:.3f})")
    
    pred_lstm, conf_lstm = predict_lstm(text)
    print(f"LSTM: {pred_lstm} ({conf_lstm:.3f})")

# =============================================================================
# 17. RINGKASAN FINAL PROYEK
# =============================================================================

# 17.1. Summary lengkap proyek
print(f"\n--- Final Summary ---")
unique_labels = sorted(y.unique())
print(f"📊 Dataset: {len(df)} samples")
print(f"🏷️ Classes: {', '.join(unique_labels)}")
print(f"🏆 Best model: {best_model_info['Model']} - {best_model_info['Test Acc']:.4f}")
print(f"✅ Criteria: {'✅ PASSED' if len(models_above_85) >= 3 else '⚠ FAILED'}")
print("🎯 SENTIMENT ANALYSIS PROJECT COMPLETED WITH GPU ACCELERATION")

# =============================================================================
# 18. FUNGSI UPDATE REPORT.MD DENGAN HASIL AKTUAL
# =============================================================================

def update_report_with_results():
    """Update report.md file dengan hasil aktual dari analisis"""
    
    # 18.1. Path ke file report
    report_path = os.path.join(current_dir, 'report.md')
    
    # 18.2. Baca file report
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 18.3. Hitung statistik dataset
        dataset_size = len(df)
        sentiment_counts = df['polarity'].value_counts()
        total_samples = sentiment_counts.sum()
        
        positive_count = sentiment_counts.get('positive', 0)
        negative_count = sentiment_counts.get('negative', 0)
        neutral_count = sentiment_counts.get('neutral', 0)
        
        positive_percent = round((positive_count / total_samples) * 100, 1)
        negative_percent = round((negative_count / total_samples) * 100, 1)
        neutral_percent = round((neutral_count / total_samples) * 100, 1)
        
        # 18.4. Update placeholders dengan nilai aktual
        replacements = {
            '{DATASET_SIZE}': f"{dataset_size:,}",
            '{POSITIVE_COUNT}': f"{positive_count:,}",
            '{NEGATIVE_COUNT}': f"{negative_count:,}",
            '{NEUTRAL_COUNT}': f"{neutral_count:,}",
            '{POSITIVE_PERCENT}': f"{positive_percent}",
            '{NEGATIVE_PERCENT}': f"{negative_percent}",
            '{NEUTRAL_PERCENT}': f"{neutral_percent}",
            
            # Model accuracies
            '{LR_TRAIN_ACC}': f"{train_acc1:.4f}",
            '{LR_TEST_ACC}': f"{test_acc1:.4f}",
            '{LR_TRAIN_ACC_PERCENT}': f"{train_acc1*100:.2f}",
            '{LR_TEST_ACC_PERCENT}': f"{test_acc1*100:.2f}",
            '{LR_STATUS}': "✅ PASS" if test_acc1 >= 0.85 else "❌ FAIL",
            
            '{RF_TFIDF_TRAIN_ACC}': f"{train_acc2:.4f}",
            '{RF_TFIDF_TEST_ACC}': f"{test_acc2:.4f}",
            '{RF_TFIDF_TRAIN_ACC_PERCENT}': f"{train_acc2*100:.2f}",
            '{RF_TFIDF_TEST_ACC_PERCENT}': f"{test_acc2*100:.2f}",
            '{RF_TFIDF_STATUS}': "✅ PASS" if test_acc2 >= 0.85 else "❌ FAIL",
            
            '{RF_BOW_TRAIN_ACC}': f"{train_acc3:.4f}",
            '{RF_BOW_TEST_ACC}': f"{test_acc3:.4f}",
            '{RF_BOW_TRAIN_ACC_PERCENT}': f"{train_acc3*100:.2f}",
            '{RF_BOW_TEST_ACC_PERCENT}': f"{test_acc3*100:.2f}",
            '{RF_BOW_STATUS}': "✅ PASS" if test_acc3 >= 0.85 else "❌ FAIL",
            
            '{LSTM_TRAIN_ACC}': f"{train_acc_lstm:.4f}",
            '{LSTM_TEST_ACC}': f"{test_acc_lstm:.4f}",
            '{LSTM_TRAIN_ACC_PERCENT}': f"{train_acc_lstm*100:.2f}",
            '{LSTM_TEST_ACC_PERCENT}': f"{test_acc_lstm*100:.2f}",
            '{LSTM_STATUS}': "✅ PASS" if test_acc_lstm >= 0.85 else "❌ FAIL",
            
            # Best model info
            '{BEST_MODEL}': best_model_info['Model'],
            '{BEST_ACCURACY}': f"{best_model_info['Test Acc']:.4f}",
            
            # Criteria validation
            '{MODELS_ABOVE_85_COUNT}': str(len(models_above_85)),
            '{FINAL_STATUS}': "✅ PASSED" if len(models_above_85) >= 3 else "❌ FAILED"
        }
        
        # 18.5. Replace semua placeholders
        for placeholder, value in replacements.items():
            content = content.replace(placeholder, str(value))
        
        # 18.6. Tulis kembali file report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Report updated successfully: {report_path}")
        
        # 18.7. Summary update yang dilakukan
        print(f"\n--- Report Update Summary ---")
        print(f"📊 Dataset size: {dataset_size:,} samples")
        print(f"📈 Best model: {best_model_info['Model']} ({best_model_info['Test Acc']:.4f})")
        print(f"✅ Models above 85%: {len(models_above_85)}/4")
        print(f"🎯 Final status: {'PASSED' if len(models_above_85) >= 3 else 'FAILED'}")
        
    except FileNotFoundError:
        print(f"❌ Report file not found: {report_path}")
    except Exception as e:
        print(f"❌ Error updating report: {e}")

# 18.8. Panggil fungsi update report
print(f"\n🔄 Updating report.md with actual results...")
update_report_with_results()