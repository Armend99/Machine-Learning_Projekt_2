import random
import streamlit as st
from datasets import load_dataset
from qa_logik import load_model, predict_gold_base_post
import pandas as pd
import plotly.express as px

# -----------------------------
# Konfiguration & Styling
# -----------------------------
st.set_page_config(page_title="GermanQuAD QA Demo", layout="wide")

# CSS für einheitliches Design
st.markdown("""
<style>
    .stButton > button {
        background-color: #0d2a3f !important;
        color: white !important;
        border-radius: 25px !important;
        padding: 0.5rem 2rem !important;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        background-color: #1a4a6e !important;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #0d2a3f;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

MODEL_DIR = "Model/qa_gelectra"
DATASET_NAME = "deepset/germanquad"
DATASET_SPLIT = "test"


# -----------------------------
# Erklärungen (Glossar & Metriken)
# -----------------------------
def render_explanations():
    st.title("🔍 Analyse & Evaluation")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Pipeline-Schritte (Vorbereitung)")
        st.info("""
        Bevor das Modell die Antwort findet, durchläuft der Text diese Schritte:
        * **Tokenization:** Zerlegung des Textes in Zahlenwerte (Tokens).
        * **Lowercasing:** Umwandlung in Kleinschreibung zur Normalisierung.
        * **Embedding:** Umwandlung der Tokens in mathematische Vektoren (Zahlenräume).
        * **Stemming (optional):** Reduktion von Wörtern auf ihren Stamm.
        """)

    with col_b:
        st.markdown("### Bewertungs-Metriken")
        st.info("""
        Wie messen wir Erfolg? In dieser Demo nutzen wir:
        * **Exact Match (EM):** Ist die Antwort zu 100% identisch mit der Gold-Antwort?
        * **F1-Score:** Misst die Wort-Überlappung zwischen Modell und Gold.
        * *Warum keine anderen?* Metriken wie ROUGE oder BLEU sind für freie Texte (Generierung) gedacht. In der extraktiven QA zählt nur die exakte Spanne.
        """)


# -----------------------------
# Modell & Daten laden
# -----------------------------
@st.cache_resource
def get_model():
    return load_model(MODEL_DIR)


@st.cache_data
def get_dataset():
    return load_dataset(DATASET_NAME, revision="refs/convert/parquet")


def sample_indices(n: int, split_len: int):
    return random.sample(range(split_len), min(n, split_len))


# -----------------------------
# UI: Sidebar
# -----------------------------
st.sidebar.header("⚙️ Modell-Parameter")

n_best_size = st.sidebar.slider(
    "n_best_size", 5, 60, 30, 5,
    help="Anzahl der Kandidaten für Start- und End-Tokens."
)
max_answer_length = st.sidebar.slider(
    "max_answer_length", 10, 120, 50, 5,
    help="Maximale Länge der extrahierten Antwort."
)
max_seq_length = st.sidebar.selectbox(
    "max_seq_length", [256, 384, 512], 1,
    help="Maximale Eingabelänge für das Modell."
)
doc_stride = st.sidebar.selectbox(
    "doc_stride", [64, 128, 192, 256], 1,
    help="Überlappung zwischen den Text-Chunks."
)

# -----------------------------
# Hauptseite
# -----------------------------
render_explanations()

st.markdown("---")
st.header("🎮 Interaktive Demo")
st.write("Wähle Beispiele aus dem GermanQuAD Datensatz und vergleiche die Vorhersage-Strategien.")

tokenizer, model, device = get_model()
ds = get_dataset()
split = ds[DATASET_SPLIT]

if "indices" not in st.session_state:
    st.session_state.indices = sample_indices(5, len(split))

if st.button("🔄 Neue zufällige Fragen laden", use_container_width=True):
    st.session_state.indices = sample_indices(5, len(split))
    st.rerun()

# Anzeige der Beispiele
for idx in st.session_state.indices:
    ex = split[idx]
    question, context = ex["question"], ex["context"]
    gold_text = ex["answers"]["text"][0] if ex.get("answers") else ""

    with st.expander(f"📍 Frage: {question}", expanded=False):
        st.markdown(f"**Kontext:**\n<div style='font-size: 14px; color: #555;'>{context}</div>", unsafe_allow_html=True)
        st.write("---")

        # Vorhersage aufrufen
        res = predict_gold_base_post(tokenizer, model, device, question, context, gold_text,
                                     n_best_size, max_answer_length, max_seq_length, doc_stride)

        # Spalten für die Textantworten
        cA, cB, cC = st.columns(3)
        with cA:
            st.markdown("🥇 **Gold (Referenz)**")
            st.success(res["gold"])
        with cB:
            st.markdown("🤖 **Base (Rohwert)**")
            st.warning(res["base"] if res["base"] else "Keine Antwort")
        with cC:
            st.markdown("✨ **Post (Optimiert)**")
            st.info(res["post"] if res["post"] else "Keine Antwort")

        st.markdown("---")
        st.subheader("📊 Vergleich der Qualitäts-Metriken")
        st.write("Wie gut stimmen die Vorhersagen mit der Gold-Antwort überein? (1.0 = Perfekt)")

        # 1. Metriken berechnen
        metrics_data = {
            "Methode": ["Base (Rohwert)", "Post (Optimiert)"],
            "Exact Match": [
                1.0 if str(res["base"]).strip().lower() == str(res["gold"]).strip().lower() else 0.0,
                1.0 if str(res["post"]).strip().lower() == str(res["gold"]).strip().lower() else 0.0
            ],
            "Wort-Überlappung (F1)": [
                len(set(str(res["base"]).lower().split()) & set(str(res["gold"]).lower().split())) / max(
                    len(set(str(res["gold"]).lower().split())), 1),
                len(set(str(res["post"]).lower().split()) & set(str(res["gold"]).lower().split())) / max(
                    len(set(res["gold"].lower().split())), 1)
            ]
        }

        df_metrics = pd.DataFrame(metrics_data)
        df_plot = df_metrics.melt(id_vars="Methode", var_name="Metrik", value_name="Score")

        # 2. Balkendiagramm erstellen
        fig = px.bar(
            df_plot,
            x="Score",
            y="Methode",
            color="Metrik",
            barmode="group",
            orientation="h",
            title="Performance-Vergleich: EM & F1",
            color_discrete_map={
                "Exact Match": "#0d2a3f",
                "Wort-Überlappung (F1)": "#1f77b4"
            },
            range_x=[0, 1]
        )

        fig.update_layout(
            height=250,
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # WICHTIG: Eindeutiger Key fixiert den Streamlit-Fehler
        st.plotly_chart(fig, use_container_width=True, key=f"metrics_chart_{idx}")

        st.caption("ℹ️ **Exact Match:** Antwort ist identisch. **F1:** Anteil der korrekten Wörter in der Antwort.")

st.markdown("---")
st.caption("Machine Learning Projekt 2026 - Fokus: GermanQuAD Evaluation")