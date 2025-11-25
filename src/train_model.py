# src/train_model.py  ← REEMPLAZA TODO EL ARCHIVO CON ESTO
import pandas as pd
import csv
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib

# RUTAS FIJAS Y SEGURAS (funcionan siempre)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "AI_Human.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Crear carpeta models si no existe
os.makedirs(MODELS_DIR, exist_ok=True)

print("Cargando dataset...")
df = pd.read_csv(DATA_PATH, quoting=csv.QUOTE_MINIMAL, escapechar='\\', engine='python', on_bad_lines='skip')
df['generated'] = pd.to_numeric(df['generated'], errors='coerce')
df.dropna(subset=['generated'], inplace=True)
df['generated'] = df['generated'].astype(int)

print(f"Dataset cargado: {df.shape[0]} filas")
df = df.sample(n=50000, random_state=42)
print("Muestra tomada: 50,000 filas")

vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X = vectorizer.fit_transform(df['text']).toarray()
y = df['generated'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=8, batch_size=64, validation_split=0.2, verbose=1)

# Evaluación
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred, target_names=['Humano', 'IA']))

# GUARDADO SEGURO
model_path = os.path.join(MODELS_DIR, "detector_ia_model.h5")
vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")

model.save(model_path)
joblib.dump(vectorizer, vectorizer_path)

print(f"\n¡ÉXITO TOTAL!")
print(f"Modelo guardado en: {model_path}")
print(f"Vectorizador guardado en: {vectorizer_path}")
print("YA PUEDES EJECUTAR: python src/app.py")