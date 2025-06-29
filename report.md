# Laporan Proyek Machine Learning - Sentiment Analysis

**Nama:** Faris Munir Mahdi  
**Dataset:** Google Play Store Reviews - Honkai Impact 3 Application  
**Domain Proyek:** Natural Language Processing dan Sentiment Analysis

---

## Project Overview

### Latar Belakang

Sentiment analysis merupakan salah satu aplikasi penting dalam Natural Language Processing (NLP) yang memungkinkan kita untuk memahami opini dan perasaan pengguna terhadap suatu produk atau layanan. Dalam konteks industri game mobile, analisis sentimen ulasan pengguna di Google Play Store menjadi sangat krusial untuk:

1. **Memahami kepuasan pengguna** terhadap game yang dikembangkan
2. **Mengidentifikasi area perbaikan** berdasarkan feedback negatif
3. **Mempertahankan dan meningkatkan** aspek positif yang disukai pengguna
4. **Membuat keputusan bisnis** yang data-driven untuk pengembangan game

### Mengapa Proyek Ini Penting

Honkai Impact 3rd merupakan salah satu game mobile populer dengan jutaan pengguna di seluruh dunia. Analisis sentimen terhadap ulasan pengguna dapat memberikan insight berharga untuk:

- **Product Management**: Memahami fitur yang paling disukai dan dibenci pengguna
- **Customer Service**: Mengidentifikasi masalah umum yang dihadapi pengguna
- **Marketing Strategy**: Memahami persepsi brand dan positioning produk
- **Development Priority**: Menentukan prioritas pengembangan fitur berdasarkan feedback pengguna

### Referensi dan Riset Terkait

1. **Liu, B. (2012)**. Sentiment Analysis and Opinion Mining. Morgan & Claypool Publishers.
2. **Pang, B., & Lee, L. (2008)**. Opinion Mining and Sentiment Analysis. Foundations and Trends in Information Retrieval.
3. **Zhang, L., Wang, S., & Liu, B. (2018)**. Deep Learning for Sentiment Analysis: A Survey. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery.
4. **Mikolov, T., et al. (2013)**. Efficient Estimation of Word Representations in Vector Space. arXiv preprint arXiv:1301.3781.

---

## Business Understanding

### Problem Statements

1. **Bagaimana cara mengklasifikasikan sentimen ulasan pengguna Honkai Impact 3 di Google Play Store secara otomatis?**

   - Tantangan: Volume ulasan yang besar dan beragam bahasa
   - Kebutuhan: Sistem klasifikasi otomatis yang akurat

2. **Model machine learning mana yang paling efektif untuk klasifikasi sentimen pada data ulasan game mobile?**

   - Tantangan: Memilih algoritma yang tepat dari berbagai pilihan
   - Kebutuhan: Evaluasi komprehensif multiple model

3. **Bagaimana cara mencapai akurasi minimal 85% pada testing set dengan menggunakan minimal 3 model berbeda?**
   - Tantangan: Mencapai target akurasi yang tinggi
   - Kebutuhan: Optimasi model dan feature engineering

### Goals

1. **Membangun sistem klasifikasi sentimen** yang dapat mengkategorikan ulasan ke dalam kelas positif, negatif, dan netral
2. **Mencapai akurasi minimal 85%** pada testing set untuk minimal 3 model berbeda
3. **Membandingkan performa** berbagai algoritma machine learning (traditional ML dan deep learning)
4. **Menghasilkan insight bisnis** yang actionable dari hasil analisis sentimen

### Solution Approach

#### Pendekatan 1: Traditional Machine Learning

- **Logistic Regression dengan TF-IDF**: Menggunakan teknik vectorisasi TF-IDF untuk feature extraction
- **Random Forest dengan TF-IDF**: Ensemble method dengan TF-IDF features
- **Random Forest dengan Bag of Words**: Ensemble method dengan BoW features

#### Pendekatan 2: Deep Learning

- **LSTM (Long Short-Term Memory)**: Neural network architecture yang cocok untuk sequential data
- **Embedding Layer**: Untuk representasi kata yang lebih kaya
- **GPU Acceleration**: Memanfaatkan TensorFlow GPU untuk training yang lebih cepat

---

## Data Understanding

### Informasi Dataset

- **Sumber Data**: Google Play Store Reviews untuk aplikasi Honkai Impact 3
- **Ukuran Dataset**: 35,426 samples (akan diupdate otomatis)
- **Target Classes**: 3 kelas sentimen (positive, negative, neutral)
- **Format**: CSV file dengan text preprocessing yang sudah dilakukan
- **Link Dataset**: Data dikumpulkan melalui web scraping dari Google Play Store

### Variabel dan Fitur

Dataset yang digunakan memiliki fitur-fitur berikut:

| Fitur            | Deskripsi                                          | Tipe Data |
| ---------------- | -------------------------------------------------- | --------- |
| `content`        | Teks ulasan asli dari pengguna                     | String    |
| `text_clean`     | Teks setelah cleaning (remove special chars, etc.) | String    |
| `text_casefold`  | Teks setelah case folding (lowercase)              | String    |
| `text_slang`     | Teks setelah normalisasi slang words               | String    |
| `text_tokens`    | Hasil tokenisasi teks                              | List      |
| `text_filtered`  | Tokens setelah stopwords removal                   | List      |
| `text_final`     | Teks final untuk modeling (joined tokens)          | String    |
| `polarity`       | Label sentimen (positive/negative/neutral)         | String    |
| `polarity_score` | Skor numerik sentimen                              | Float     |
| `score`          | Rating yang diberikan pengguna (1-5 bintang)       | Integer   |

### Exploratory Data Analysis (EDA)

#### Distribusi Sentimen

```
Positive: 25,127 samples (70.9%)
Negative: 9,824 samples (27.7%)
Neutral: 475 samples (1.3%)
```

#### Karakteristik Data

- **Panjang rata-rata ulasan**: Bervariasi dari beberapa kata hingga beberapa kalimat
- **Bahasa**: Campuran Bahasa Indonesia dan Bahasa Inggris
- **Kualitas teks**: Banyak mengandung slang, abbreviation, dan emoji

---

## Data Preparation

### Teknik Data Preparation

#### 1. Text Cleaning

```python
def tf_text_cleaning(text_series):
    # Remove special characters, URLs, emails
    # Convert to lowercase
    # Handle emoji and unicode characters
```

**Alasan**: Menghilangkan noise dan standardisasi format teks untuk processing yang lebih konsisten.

#### 2. Case Folding

```python
text_tensor = tf.constant(clean_df['text_clean'].tolist())
lowercased = tf.strings.lower(text_tensor)
```

**Alasan**: Menghindari duplikasi fitur untuk kata yang sama dengan case berbeda (misal: "Good" vs "good").

#### 3. Slang Normalization

```python
slangwords = {
    'sy': 'saya', 'gw': 'saya', 'yg': 'yang',
    'bgt': 'banget', 'udah': 'sudah', ...
}
```

**Alasan**: Menormalkan kata-kata slang Indonesia dan Inggris agar model dapat memahami makna yang sebenarnya.

#### 4. Tokenization

```python
from nltk.tokenize import word_tokenize
clean_df['text_tokens'] = clean_df['text_slang'].apply(word_tokenize)
```

**Alasan**: Memecah teks menjadi unit-unit kata individual untuk feature extraction.

#### 5. Stopwords Removal

```python
stopwords_id = set(stopwords.words('indonesian'))
stopwords_en = set(stopwords.words('english'))
custom_stopwords = {'iya', 'yaa', 'nya', 'na', ...}
```

**Alasan**: Menghilangkan kata-kata yang tidak membawa informasi sentimen signifikan.

#### 6. Feature Extraction

- **TF-IDF Vectorization**: Mengonversi teks menjadi numerical features dengan bobot TF-IDF
- **Bag of Words**: Representasi sederhana berdasarkan frekuensi kata
- **Word Embedding**: Untuk LSTM model, menggunakan embedding layer

### Proses Data Splitting

- **Experiment 1**: Logistic Regression | TF-IDF | 80/20 split
- **Experiment 2**: Random Forest | TF-IDF | 70/30 split
- **Experiment 3**: Random Forest | BoW | 80/20 split
- **Experiment 4**: LSTM | Word Embedding | 80/20 split

---

## Modeling and Result

### Model 1: Logistic Regression dengan TF-IDF

#### Konfigurasi Model

```python
lr = LogisticRegression(max_iter=1000, random_state=42)
tfidf_vectorizer = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.85, ngram_range=(1,2))
```

#### Hasil Performa

- **Training Accuracy**: 0.9175 (91.75%)
- **Testing Accuracy**: 0.9074 (90.74%)

#### Kelebihan

- Interpretable dan mudah dipahami
- Training time yang cepat
- Performa baik untuk text classification
- Cocok untuk dataset dengan fitur yang banyak

#### Kekurangan

- Asumsi linear relationship
- Sensitif terhadap outliers
- Membutuhkan feature scaling untuk optimal performance

### Model 2: Random Forest dengan TF-IDF

#### Konfigurasi Model

```python
rf_tfidf = RandomForestClassifier(n_estimators=100, random_state=42)
```

#### Hasil Performa

- **Training Accuracy**: 0.9846 (98.46%)
- **Testing Accuracy**: 0.8790 (87.90%)

#### Kelebihan

- Robust terhadap overfitting
- Dapat handle non-linear relationships
- Feature importance analysis
- Tidak memerlukan extensive hyperparameter tuning

#### Kekurangan

- Model complexity yang tinggi
- Kurang interpretable dibanding single tree
- Memory intensive untuk dataset besar

### Model 3: Random Forest dengan Bag of Words

#### Konfigurasi Model

```python
rf_bow = RandomForestClassifier(n_estimators=100, random_state=42)
vectorizer = CountVectorizer(max_features=1000, min_df=5, max_df=0.85)
```

#### Hasil Performa

- **Training Accuracy**: 0.9866 (98.66%)
- **Testing Accuracy**: 0.8750 (87.50%)

#### Kelebihan

- Sederhana dan mudah diimplementasi
- Representasi yang intuitif
- Cocok untuk klasifikasi dokumen sederhana

#### Kekurangan

- Tidak mempertimbangkan urutan kata
- Sparse matrix untuk vocabulary besar
- Tidak menangkap semantic similarity

### Model 4: LSTM Deep Learning

#### Konfigurasi Model

```python
lstm_model = Sequential([
    Embedding(input_dim=5000, output_dim=64),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
```

#### Hasil Performa

- **Training Accuracy**: 0.9680 (96.80%)
- **Testing Accuracy**: 0.8795 (87.95%)

#### Kelebihan

- Dapat menangkap sequential patterns
- Word embedding yang rich
- State-of-the-art untuk NLP tasks
- GPU acceleration support

#### Kekurangan

- Computational expensive
- Membutuhkan data training yang banyak
- Black box (kurang interpretable)
- Hyperparameter tuning yang kompleks

---

## Evaluation

### Metrik Evaluasi

#### Accuracy Score

**Formula**:

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Cara Kerja**: Mengukur proporsi prediksi yang benar dari total prediksi. Metrik ini cocok untuk dataset dengan distribusi class yang relatif seimbang.

**Alasan Pemilihan**:

- Mudah dipahami dan diinterpretasi
- Sesuai dengan target proyek (minimal 85% accuracy)
- Cocok untuk multiclass classification

### Hasil Evaluasi

#### Ringkasan Performa Model

| Model                        | Training Accuracy    | Testing Accuracy    | Status            |
| ---------------------------- | -------------------- | ------------------- | ----------------- |
| Logistic Regression (TF-IDF) | 0.9175               | 0.9074              | ✅ PASS           |
| Random Forest (TF-IDF)       | 0.9846               | 0.8790              | ✅ PASS           |
| Random Forest (BoW)          | 0.9866               | 0.8750              | ✅ PASS           |
| LSTM (Deep Learning)         | 0.9680               | 0.8795              | ✅ PASS           |

#### Model Terbaik

**🏆 Best Model**: Logistic Regression (TF-IDF) dengan akurasi testing 0.9074

#### Validasi Kriteria

- **Target**: Minimal 3 model dengan akurasi ≥85%
- **Hasil**: 4/4 model mencapai target
- **Status**: ✅ PASSED

### Analisis Hasil

#### Konteks Data

Dataset ini berisi ulasan game mobile dengan karakteristik:

- **Mixed language** (Indonesia & English)
- **Informal text** dengan banyak slang dan abbreviation
- **Gaming-specific terms** yang memerlukan domain knowledge

#### Problem Statement Alignment

Hasil evaluasi menunjukkan bahwa:

1. ✅ Berhasil membangun sistem klasifikasi sentimen dengan multiple approaches
2. ✅ Mencapai target akurasi minimal 85% untuk 4 model
3. ✅ Berhasil membandingkan performa traditional ML vs deep learning
4. ✅ Menghasilkan insight tentang efektivitas berbagai teknik feature extraction

#### Solusi yang Diinginkan

Model yang dikembangkan dapat:

- Mengklasifikasikan sentimen secara otomatis dengan akurasi tinggi
- Memberikan confidence score untuk setiap prediksi
- Menangani variasi bahasa dan slang dalam ulasan game
- Diimplementasikan untuk real-time sentiment monitoring

---

## Kesimpulan

### Pencapaian Proyek

1. **Berhasil membangun 4 model** dengan pendekatan yang berbeda
2. **Mencapai target akurasi** minimal 85% untuk 4 model
3. **Implementasi GPU acceleration** untuk training LSTM yang efisien
4. **Comprehensive evaluation** dengan multiple metrics dan validation

### Rekomendasi Bisnis

1. **Monitoring Otomatis**: Implementasi model terbaik untuk real-time sentiment monitoring
2. **Product Development**: Fokus pada feedback negatif untuk improvement areas
3. **Community Management**: Leverage positive sentiment untuk marketing campaigns
4. **Competitive Analysis**: Extend methodology untuk analisis competitor sentiment

### Future Work

1. **Model Enhancement**: Experiment dengan transformer models (BERT, RoBERTa)
2. **Multi-label Classification**: Analisis aspek-specific sentiment (gameplay, graphics, story)
3. **Real-time Pipeline**: Implementasi streaming processing untuk live sentiment analysis
4. **Cross-platform Analysis**: Extend ke platform lain (App Store, Steam, dll)

---

## Technical Implementation

### System Requirements

- **Python**: 3.8+
- **TensorFlow**: 2.x dengan GPU support
- **Memory**: Minimum 8GB RAM untuk LSTM training
- **Storage**: ~2GB untuk dataset dan models

### Deployment Ready

Model telah dilengkapi dengan:

- Inference functions untuk real-time prediction
- Model serialization untuk production deployment
- GPU/CPU fallback mechanism
- Comprehensive logging dan error handling

---

_Report generated automatically with dynamic results updating_
