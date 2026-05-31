import random
import json
import streamlit as st
from datasets import Dataset, DatasetDict
from qa_logik import load_model, predict_gold_base_post

# -----------------------------
# Konfiguration & Styling
# -----------------------------
st.set_page_config(page_title="GermanQuAD QA Demo", layout="wide")

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
    .answer-card {
        padding: 16px;
        border-radius: 10px;
        margin-bottom: 8px;
    }
    .gold-card { background-color: #e8f5ee; border-left: 5px solid #2a7a4f; }
    .base-card { background-color: #fefce8; border-left: 5px solid #d97706; }
    .post-card { background-color: #e8f0fa; border-left: 5px solid #1a4a6e; }
</style>
""", unsafe_allow_html=True)

MODEL_DIR     = "Model/qa_gelectra_v2"
DATASET_SPLIT = "test"

# Feste Werte (nicht mehr über Sidebar verstellbar)
N_BEST_SIZE       = 30
MAX_ANSWER_LENGTH = 50

# -----------------------------
# Erklärungen: Gold / Base / Post
# -----------------------------
def render_explanations():
    st.title("🔍 Analyse & Evaluation")

    st.markdown("### Die drei Antworten im Vergleich")
    st.write("""
    In der Demo siehst du zu jeder Frage drei Antworten. Sie haben unterschiedliche Rollen:
    """)

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("""
        <div class="answer-card gold-card">
        <strong>🥇 Gold (Referenz)</strong><br><br>
        Die <strong>korrekte Antwort</strong> aus dem GermanQuAD-Datensatz — von Menschen
        festgelegt. Sie ist der Maßstab, an dem die Vorhersagen gemessen werden.
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="answer-card base-card">
        <strong>🤖 Base (einfach)</strong><br><br>
        Die <strong>einfachste Vorhersage</strong>: Das Modell nimmt direkt den Token mit dem
        höchsten Start-Wert und den mit dem höchsten End-Wert (argmax). Schnell, aber an den
        Rändern manchmal ungenau.
        </div>
        """, unsafe_allow_html=True)

    with col_c:
        st.markdown("""
        <div class="answer-card post-card">
        <strong>✨ Post (optimiert)</strong><br><br>
        Die <strong>verfeinerte Vorhersage</strong>: Statt nur den höchsten Wert zu nehmen,
        prüft Post die besten Kandidaten und wählt die plausibelste gültige Antwortspanne.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Was bedeutet n_best_size?")
    st.info("""
    **n_best_size** ist die Anzahl der Kandidaten, die die **Post-Methode** prüft.

    * **Base** nimmt nur die *eine* beste Start- und End-Position (Top-1, argmax) — dafür braucht es kein n_best_size.
    * **Post** betrachtet die besten *n* Start- und End-Positionen (z. B. die Top-30) und prüft alle gültigen Kombinationen. So findet das Modell auch dann die richtige Spanne, wenn die einfache Top-1-Wahl danebenliegt.

    In diesem Projekt ist n_best_size fest auf **30** gesetzt. Interessant: Beim finalen Modell bringt Post kaum noch einen Vorteil gegenüber Base — das Training war präzise genug, dass die aufwändige n-best-Suche überflüssig wird.
    """)

# -----------------------------
# Modell & Daten laden
# -----------------------------
@st.cache_resource
def get_model():
    return load_model(MODEL_DIR)

@st.cache_data
def get_dataset():
    def parse(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rows = []
        for article in data["data"]:
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    rows.append({
                        "id":       qa["id"],
                        "context":  context,
                        "question": qa["question"],
                        "answers":  {
                            "text":         [a["text"] for a in qa["answers"]],
                            "answer_start": [a["answer_start"] for a in qa["answers"]]
                        }
                    })
        return rows

    return DatasetDict({
        "train": Dataset.from_list(parse("GermanQuAD_train.json")),
        "test":  Dataset.from_list(parse("GermanQuAD_test.json"))
    })

def sample_indices(n: int, split_len: int):
    return random.sample(range(split_len), min(n, split_len))

def f1_score(pred: str, gold: str) -> float:
    p = set(pred.lower().split())
    g = set(gold.lower().split())
    if not p or not g:
        return 0.0
    overlap   = p & g
    precision = len(overlap) / len(p)
    recall    = len(overlap) / len(g)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

# -----------------------------
# UI: Sidebar (nur max_seq_length und doc_stride)
# -----------------------------
st.sidebar.header("⚙️ Modell-Parameter")

max_seq_length = st.sidebar.selectbox(
    "max_seq_length", [256, 384, 512], 2,
    help="Maximale Eingabelänge für das Modell (Anzahl Tokens)."
)
doc_stride = st.sidebar.selectbox(
    "doc_stride", [64, 128, 192, 256], 1,
    help="Überlappung zwischen den Text-Chunks beim Sliding Window."
)

# -----------------------------
# Automatische Neuberechnung bei Parameter-Änderung
# -----------------------------
current_params = (max_seq_length, doc_stride)

if "last_params" not in st.session_state:
    st.session_state.last_params = current_params

if current_params != st.session_state.last_params:
    st.session_state.last_params = current_params
    st.rerun()

# -----------------------------
# Hauptseite
# -----------------------------
render_explanations()

st.markdown("---")
st.header("🎮 Interaktive Demo")
st.write("Wähle Beispiele aus dem GermanQuAD Datensatz und vergleiche die Vorhersage-Strategien.")

tokenizer, model, device = get_model()
ds    = get_dataset()
split = ds[DATASET_SPLIT]

if "indices" not in st.session_state:
    st.session_state.indices = sample_indices(5, len(split))

if st.button("🔄 Neue zufällige Fragen laden", use_container_width=True):
    st.session_state.indices = sample_indices(5, len(split))
    st.rerun()

# -----------------------------
# Offizielle Modell-Performance
# -----------------------------
st.markdown("### 📊 Offizielle Modell-Performance")
st.caption("Berechnet auf dem gesamten GermanQuAD Testset (n=2.204 Fragen)")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Exact Match (Base)", value="—")
with col2:
    st.metric(label="F1-Score (Base)", value="—")
with col3:
    st.metric(label="Exact Match (Post)", value="63.75%")
with col4:
    st.metric(label="F1-Score (Post)", value="79.54%")

st.markdown("---")

# -----------------------------
# Alle 5 Fragen vorberechnen
# -----------------------------
results = []
for idx in st.session_state.indices:
    ex        = split[idx]
    question  = ex["question"]
    context   = ex["context"]
    gold_text = ex["answers"]["text"][0] if ex.get("answers") else ""

    res = predict_gold_base_post(
        tokenizer, model, device,
        question, context, gold_text,
        N_BEST_SIZE, MAX_ANSWER_LENGTH, max_seq_length, doc_stride
    )
    results.append((idx, ex, res))

# -----------------------------
# Zusammenfassung der 5 Fragen
# -----------------------------
base_em = sum(
    1 for _, _, r in results
    if r["base"].strip().lower() == r["gold"].strip().lower()
) / len(results)

post_em = sum(
    1 for _, _, r in results
    if r["post"].strip().lower() == r["gold"].strip().lower()
) / len(results)

base_f1 = sum(f1_score(r["base"], r["gold"]) for _, _, r in results) / len(results)
post_f1 = sum(f1_score(r["post"], r["gold"]) for _, _, r in results) / len(results)

st.markdown("### 🔬 Aktuelle Stichprobe (5 Fragen)")
st.caption("Zufällig gewählte Beispiele — Werte variieren bei jedem Laden")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric(
        label="Exact Match (Base)",
        value=f"{base_em*100:.1f}%",
        delta=f"{(base_em - 0.6375)*100:.1f}% vs. Testset"
    )
with c2:
    st.metric(
        label="F1-Score (Base)",
        value=f"{base_f1*100:.1f}%",
        delta=f"{(base_f1 - 0.7954)*100:.1f}% vs. Testset"
    )
with c3:
    st.metric(
        label="Exact Match (Post)",
        value=f"{post_em*100:.1f}%",
        delta=f"{(post_em - 0.6375)*100:.1f}% vs. Testset"
    )
with c4:
    st.metric(
        label="F1-Score (Post)",
        value=f"{post_f1*100:.1f}%",
        delta=f"{(post_f1 - 0.7954)*100:.1f}% vs. Testset"
    )

st.markdown("---")

# -----------------------------
# Einzelne Fragen
# -----------------------------
for idx, ex, res in results:
    question = ex["question"]
    context  = ex["context"]

    base_correct = res["base"].strip().lower() == res["gold"].strip().lower()
    post_correct = res["post"].strip().lower() == res["gold"].strip().lower()

    with st.expander(f"📍 Frage: {question}", expanded=False):
        st.markdown(
            f"**Kontext:**\n<div style='font-size: 14px; color: #555;'>{context}</div>",
            unsafe_allow_html=True
        )
        st.write("---")

        cA, cB, cC = st.columns(3)
        with cA:
            st.markdown("🥇 **Gold (Referenz)**")
            st.success(res["gold"])
        with cB:
            st.markdown(f"🤖 **Base** {'✅' if base_correct else '❌'}")
            st.warning(res["base"] if res["base"] else "Keine Antwort")
            st.caption(f"F1: {f1_score(res['base'], res['gold'])*100:.1f}%")
        with cC:
            st.markdown(f"✨ **Post** {'✅' if post_correct else '❌'}")
            st.info(res["post"] if res["post"] else "Keine Antwort")
            st.caption(f"F1: {f1_score(res['post'], res['gold'])*100:.1f}%")

st.markdown("---")
st.caption("Machine Learning Projekt 2026 - Fokus: GermanQuAD Evaluation")