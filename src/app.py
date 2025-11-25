# src/app.py  ← REEMPLAZA TODO EL CONTENIDO CON ESTO
import gradio as gr
from tensorflow.keras.models import load_model
import joblib
import os

# Rutas absolutas seguras
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "detector_ia_model.h5")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.joblib")

print("Cargando modelo y vectorizador...")
model = load_model(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
print("¡Modelo cargado con éxito!")

def detectar_texto(texto):
    if not texto or not texto.strip():
        return "Escribe algo, parcero 😅"
    
    X = vectorizer.transform([texto]).toarray()
    prob = model.predict(X, verbose=0)[0][0]
    
    if prob > 0.5:
        return f"Esto lo escribió una IA\n\nConfianza: {prob*100:.1f}%"
    else:
        return f"Esto lo escribió un HUMANO\n\nConfianza: {(1-prob)*100:.1f}%"

# INTERFAZ 100% COMPATIBLE CON GRADIO 2025
with gr.Blocks() as demo:
    gr.Markdown("# Detector Colombiano: ¿IA o Humano?")
    gr.Markdown("### Checkpoint #7 – Estud-IA Audiovisual – Alcaldía de Medellín")
    gr.Markdown("Accuracy ~99% | Hecho con ❤️ por Brayan Steven")
    
    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                lines=10,
                placeholder="Pega aquí el texto que quieras analizar...",
                label="Texto a evaluar"
            )
        with gr.Column(scale=1):
            output = gr.Textbox(
                label="Resultado",
                lines=8,
                interactive=False
            )
    
    btn = gr.Button("Analizar texto", variant="primary", size="lg")
    btn.click(fn=detectar_texto, inputs=input_text, outputs=output)
    
    gr.Examples(
        examples=[
            ["La inteligencia artificial está revolucionando la educación y la medicina de forma exponencial."],
            ["Ayer fui al Éxito con mi mamá, compramos arepa con queso y nos tomamos un tinto en el parque."],
            ["Escribe un poema sobre el amor en la era digital con metáforas profundas."]
        ],
        inputs=input_text
    )

# Tema bonito (nuevo formato 2025)
demo.launch(theme=gr.themes.Soft())