# -*- coding: utf-8 -*-

"""
================================================================================
                    ANALISIS SENTIMEN ULASAN HONKAI IMPACT 3RD
================================================================================

📊 Dataset: Ulasan Honkai Impact 3rd dari Google Play Store (66,455 sampel)
🎯 Target: Minimal 3 model dengan akurasi testing ≥85%
"""

# =============================================================================
# 1. IMPORT LIBRARIES
# =============================================================================

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# TensorFlow imports for LSTM
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Configure TensorFlow to use GPU
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Enable GPU memory growth to avoid OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"⚠ GPU configuration error: {e}")
else:
    print("⚠ No GPU detected, using CPU")

# =============================================================================
# 2. LOAD DATASET  
# =============================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, 'datasets', 'ulasan_honkai_impact_3_processed.csv')

df = pd.read_csv(dataset_path)
print(f"Dataset loaded: {len(df)} samples")

# =============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

print(f"\n--- Dataset Info ---")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

print(f"\n--- Sentiment Distribution ---")
print(df['polarity'].value_counts())

print(f"\n--- Missing Values ---")
print(df.isnull().sum())

# =============================================================================
# 4. DATA CLEANING
# =============================================================================

print(f"\n--- Data Cleaning ---")
print(f"Before cleaning: {len(df)} samples")

df = df.dropna(subset=['text_final', 'polarity'])
df = df.drop_duplicates(subset=['text_final'])

print(f"After cleaning: {len(df)} samples")

if len(df) >= 3000:
    print(f"✓ Dataset meets criteria (≥3000 samples)")
else:
    print(f"⚠ Dataset below 3000 samples")

# =============================================================================
# 5. FEATURE AND TARGET PREPARATION
# =============================================================================

X = df['text_final']
y = df['polarity']

print(f"\n--- Feature & Target Info ---")
print(f"Text features: {len(X)}")
print(f"Target distribution:")
print(y.value_counts())

# =============================================================================
# 6. FEATURE EXTRACTION: TF-IDF
# =============================================================================

print(f"\n--- TF-IDF Vectorization ---")
tfidf_vectorizer = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.85, ngram_range=(1,2))
tfidf_matrix = tfidf_vectorizer.fit_transform(X)
X_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
print(f"TF-IDF shape: {X_tfidf.shape}")

# =============================================================================  
# 7. FEATURE EXTRACTION: BAG OF WORDS
# =============================================================================

print(f"\n--- Bag of Words Vectorization ---")
vectorizer = CountVectorizer(max_features=1000, min_df=5, max_df=0.85)
bow_matrix = vectorizer.fit_transform(X)
X_bow = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())
print(f"BoW shape: {X_bow.shape}")

# =============================================================================
# 8. DATA SPLITTING FOR EXPERIMENTS
# =============================================================================

print(f"\n--- Data Splitting ---")
# Experiment 1: Logistic Regression | TF-IDF | 80/20
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Experiment 2: Random Forest | TF-IDF | 70/30
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

# Experiment 3: Random Forest | BoW | 80/20
X_train3, X_test3, y_train3, y_test3 = train_test_split(X_bow, y, test_size=0.2, random_state=42)

all_results = []
print("Data split complete for 4 experiments")

# =============================================================================
# 9. MODELING: LOGISTIC REGRESSION (TF-IDF)
# =============================================================================

print(f"\n--- Experiment 1: Logistic Regression + TF-IDF ---")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train1, y_train1)

y_train_pred1 = lr.predict(X_train1)
y_test_pred1 = lr.predict(X_test1)

train_acc1 = accuracy_score(y_train1, y_train_pred1)
test_acc1 = accuracy_score(y_test1, y_test_pred1)

result1 = {'Model': 'Logistic Regression (TF-IDF)', 'Train Acc': train_acc1, 'Test Acc': test_acc1}
all_results.append(result1)
print(f"Train: {train_acc1:.4f}, Test: {test_acc1:.4f}")

# =============================================================================
# 10. MODELING: RANDOM FOREST (TF-IDF)
# =============================================================================

print(f"\n--- Experiment 2: Random Forest + TF-IDF ---")
rf_tfidf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_tfidf.fit(X_train2, y_train2)

y_train_pred2 = rf_tfidf.predict(X_train2)
y_test_pred2 = rf_tfidf.predict(X_test2)

train_acc2 = accuracy_score(y_train2, y_train_pred2)
test_acc2 = accuracy_score(y_test2, y_test_pred2)

result2 = {'Model': 'Random Forest (TF-IDF)', 'Train Acc': train_acc2, 'Test Acc': test_acc2}
all_results.append(result2)
print(f"Train: {train_acc2:.4f}, Test: {test_acc2:.4f}")

# =============================================================================
# 11. MODELING: RANDOM FOREST (BOW)
# =============================================================================

print(f"\n--- Experiment 3: Random Forest + BoW ---")
rf_bow = RandomForestClassifier(n_estimators=100, random_state=42)
rf_bow.fit(X_train3, y_train3)

y_train_pred3 = rf_bow.predict(X_train3)
y_test_pred3 = rf_bow.predict(X_test3)

train_acc3 = accuracy_score(y_train3, y_train_pred3)
test_acc3 = accuracy_score(y_test3, y_test_pred3)

result3 = {'Model': 'Random Forest (BoW)', 'Train Acc': train_acc3, 'Test Acc': test_acc3}
all_results.append(result3)
print(f"Train: {train_acc3:.4f}, Test: {test_acc3:.4f}")

# =============================================================================
# 12. MODELING: LSTM (DEEP LEARNING)
# =============================================================================

print(f"\n--- Experiment 4: LSTM Deep Learning ---")

# Prepare data for LSTM
X_text = df['text_final']
y_text = df['polarity']

# Encode labels to numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_text)
num_classes = len(label_encoder.classes_)

# Tokenization with 5000 most frequent words
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(X_text)

# Convert texts to sequences
sequences = tokenizer.texts_to_sequences(X_text)

# Pad sequences to uniform length of 20
X_padded = pad_sequences(sequences, maxlen=20, padding='post')

# One-hot encode labels for categorical crossentropy
y_categorical = to_categorical(y_encoded, num_classes=num_classes)

# Split data for LSTM (80/20)
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
    X_padded, y_categorical, test_size=0.2, random_state=42
)

# Build LSTM architecture
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    lstm_model = Sequential()
    lstm_model.add(Embedding(input_dim=5000, output_dim=64))  # Embedding layer
    lstm_model.add(LSTM(64, return_sequences=False))         # LSTM layer
    lstm_model.add(Dropout(0.5))                             # Dropout for regularization
    lstm_model.add(Dense(num_classes, activation='softmax')) # Output layer

# Compile model
lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(f"Model built on: {'GPU' if gpus else 'CPU'}")
print("LSTM Model Summary:")
lstm_model.summary()

# Train model
print("Training LSTM model...")
print(f"Training on: {'GPU' if gpus else 'CPU'}")

# Use GPU for training if available
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    history = lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=64, 
                            validation_split=0.1, verbose=1)

# Make predictions
y_train_pred_lstm = lstm_model.predict(X_train_lstm)
y_test_pred_lstm = lstm_model.predict(X_test_lstm)

# Convert predictions back to class labels
y_train_pred_labels = np.argmax(y_train_pred_lstm, axis=1)
y_test_pred_labels = np.argmax(y_test_pred_lstm, axis=1)
y_train_true_labels = np.argmax(y_train_lstm, axis=1)
y_test_true_labels = np.argmax(y_test_lstm, axis=1)

# Calculate accuracies
train_acc_lstm = accuracy_score(y_train_true_labels, y_train_pred_labels)
test_acc_lstm = accuracy_score(y_test_true_labels, y_test_pred_labels)

result4 = {'Model': 'LSTM (Deep Learning)', 'Train Acc': train_acc_lstm, 'Test Acc': test_acc_lstm}
all_results.append(result4)
print(f"Train: {train_acc_lstm:.4f}, Test: {test_acc_lstm:.4f}")

# =============================================================================
# 13. MODEL EVALUATION AND COMPARISON
# =============================================================================

print(f"\n--- Model Performance Summary ---")
accuracy_df_final = pd.DataFrame(all_results).sort_values(by='Test Acc', ascending=False)
print(accuracy_df_final.to_string(index=False))

# =============================================================================
# 14. SUBMISSION CRITERIA VALIDATION
# =============================================================================

print(f"\n--- Submission Criteria Check ---")
models_above_85 = accuracy_df_final[accuracy_df_final['Test Acc'] >= 0.85]
print(f"Models with ≥85% accuracy: {len(models_above_85)}/4")

if len(models_above_85) >= 3:
    print("✓ PASSED: ≥3 models with 85%+ accuracy")
else:
    print("⚠ FAILED: Need ≥3 models with 85%+ accuracy")

best_model_info = accuracy_df_final.iloc[0]
print(f"Best model: {best_model_info['Model']} ({best_model_info['Test Acc']:.4f})")

models_above_92 = accuracy_df_final[accuracy_df_final['Test Acc'] >= 0.92]
if len(models_above_92) >= 1:
    print("✓ BONUS: ≥1 model with 92%+ accuracy")

# =============================================================================
# 15. INFERENCE FUNCTION
# =============================================================================

def predict_sentiment(text, model, vectorizer):
    text_vector = vectorizer.transform([text])
    pred_class = model.predict(text_vector)[0]
    
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(text_vector)[0]
        confidence = max(probs)
    else:
        confidence = 1.0
    
    return pred_class, confidence

# LSTM prediction function
def predict_lstm(text):
    """Predict sentiment using LSTM model"""
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=20, padding='post')
    probs = lstm_model.predict(padded)
    pred_class = np.argmax(probs, axis=1)[0]
    confidence = max(probs[0])
    return label_encoder.inverse_transform([pred_class])[0], confidence

# =============================================================================
# 16. INFERENCE TESTING
# =============================================================================

print(f"\n--- Inference Testing ---")

test_texts = [
    "Game ini sangat bagus dan menyenangkan!",
    "Aplikasi sering crash, sangat mengecewakan",
    "Grafik bagus tapi gameplay biasa saja",
    "Honkai Impact 3rd game terbaik!",
    "Bug banyak sekali, tidak recommended"
]

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

print(f"\n--- Final Summary ---")
unique_labels = sorted(y.unique())
print(f"Dataset: {len(df)} samples")
print(f"Classes: {', '.join(unique_labels)}")
print(f"Best model: {best_model_info['Model']} - {best_model_info['Test Acc']:.4f}")
print(f"Criteria: {'✓ PASSED' if len(models_above_85) >= 3 else '⚠ FAILED'}")