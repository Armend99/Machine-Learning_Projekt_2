import streamlit as st
import base64

def img_to_data_uri(path: str) -> str:
    try:
        with open(path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("utf-8")
        if path.lower().endswith(".png"):
            mime = "image/png"
        else:
            mime = "image/jpeg"
        return f"data:{mime};base64,{b64}"
    except:
        return ""

st.set_page_config(page_title="Projekt: Machine Learning", layout="wide")

# -----------------------------
# Assets
# -----------------------------
HERO_PATH = "Assets/Bild_1.jpg"
HERO_DATA_URI = img_to_data_uri(HERO_PATH)
INFO_BG_PATH = "Assets/2_Bild.jpeg"
INFO_BG_URI  = img_to_data_uri(INFO_BG_PATH)

# -----------------------------
# Minimal CSS für Landingpage
# -----------------------------
st.markdown(f"""
<style>
.block-container {{ 
    padding-top: 0rem !important; 
    padding-bottom: 3rem; 
    max-width: 100%;
    padding-left: 0rem;
    padding-right: 0rem;
}}

/* --- HERO Sektion --- */
.hero {{
  height: 520px;
  overflow: hidden;
  background-image: url("{HERO_DATA_URI}");
  background-size: cover;
  background-attachment: fixed;
  background-position: center;
  position: relative;
}}

.hero::before {{
  content: "";
  position: absolute;
  inset: 0;
  background: rgba(0,0,0,0.18);
}}

.hero-card {{
  position: absolute;
  left: 50%;
  bottom: 10px;
  transform: translateX(-50%);
  width: min(900px, 78%);
  background: rgba(255,255,255,0.96);
  border-radius: 4px;
  padding: 22px 20px 18px 20px;
  text-align: center;
  z-index: 2;
  box-shadow: 0 14px 32px rgba(0,0,0,0.18);
}}

.hero-title {{
  font-size: 54px;
  letter-spacing: 0.08em;
  margin: 0;
  font-weight: 500;
  color: #0d2a3f;
  text-transform: uppercase;
}}

/* --- Kachel-Zentrierung & Versatz --- */
[data-testid="stVerticalBlock"] > div:has(.wix-col) {{
    margin-top: -180px !important;
}}

.wix-grid {{
  max-width: 1200px;
  margin: 0 auto;
  background: white;
  position: relative;
  z-index: 10;
  border: 1px solid rgba(0,0,0,0.15);
}}

.wix-col {{
  padding: 20px 44px 50px 44px;
  text-align: center;
  min-height: 380px;
}}
/* --- Optimierung der Kachel-Überschriften --- */
.wix-title ,.info-title{{
  text-align: center !important;  /* Zwingt den Text in die Mitte */
  width: 100% !important;         /* Nutzt die volle Breite der Kachel */
  display: block !important;
  font-size: 38px !important;     /* Etwas größer (vorher 34px) */
  font-weight: 800 !important;     /* Richtig fett (800 statt 500) */
  letter-spacing: 0.15em !important;
  color: #0d2a3f !important;      /* Passend zum dunklen Blau der Buttons */
  margin: 0 auto 20px auto !important;
  padding-top: 10px !important;
}}

/* Stellt sicher, dass auch der Beschreibungstext darunter zentriert bleibt */
.wix-text {{
  text-align: center !important;
  margin: 0 auto !important;
  font-size: 15px;
  line-height: 1.8;
  opacity: 0.85;
}}


/* --- INFO Parallax Section --- */
.info-section {{
  width: 100%;
  padding: 100px 0;
  background-image: url("{INFO_BG_URI}");
  background-size: cover;
  background-position: center;
  background-attachment: fixed;
  position: relative;
}}

.info-section::before {{
  content:"";
  position:absolute;
  inset:0;
  background: rgba(210, 230, 242, 0.72);
}}

.info-content {{
  position: relative;
  z-index: 2;
  max-width: 1100px;
  margin: 0 auto;
  text-align: center;
}}

/* --- BUTTON GLOBAL STYLING (Kacheln & Parallax) --- */
/* Dies gilt für st.button, st.link_button UND st.download_button */
.stButton, .stLinkButton, .stDownloadButton {{
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
    margin-top: 25px !important; /* Abstand nach oben */
}}

.stButton > button, .stLinkButton > a, .stDownloadButton > button {{
    background-color: #0d2a3f !important;
    color: white !important;
    border-radius: 25px !important;
    border: none !important;
    padding: 0.6rem 2.2rem !important;
    width: auto !important;
    min-width: 180px;
    font-weight: 600 !important;
    text-decoration: none !important;
    display: inline-flex !important;
    transition: background-color 0.3s ease, transform 0.2s ease !important;
}}

.stButton > button:hover, .stLinkButton > a:hover, .stDownloadButton > button:hover {{
    background-color: #1a4a6e !important;
    transform: scale(1.03);
    color: white !important;
}}

/* Fix für Spalten-Ausrichtung */
[data-testid="stColumn"] [data-testid="stVerticalBlock"] {{
    align-items: center !important;
}}
</style>
""", unsafe_allow_html=True)



# -----------------------------
# RENDER: HERO
# -----------------------------
st.markdown('<div class="hero-shell"><div class="hero">', unsafe_allow_html=True)
st.markdown("""
  <div class="hero-card">
    <div class="hero-kicker">Projekt: Machine Learning</div>
    <div class="hero-title">Fragen und Antwort System</div>
  </div>
""", unsafe_allow_html=True)
st.markdown('</div></div>', unsafe_allow_html=True)

# -----------------------------
# RENDER: MEHR INFOS KACHELN
# -----------------------------
st.markdown('<div class="wix-grid">', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)

with c1:
    with c1:
        st.markdown('<div class="wix-col">', unsafe_allow_html=True)
        st.markdown('<div class="wix-title">MODELL</div>', unsafe_allow_html=True)
        st.markdown('<div class="wix-text">Extratice QA: Das System beantwrtet Fragen, indem es eine passende Textspanne aus dem Kontext extrahiert. Also keine freie Textgenerierung</div>', unsafe_allow_html=True)
        st.markdown('<div class="wix-btn-wrap">', unsafe_allow_html=True)

        # NUR DIESE ZEILE ÄNDERN:
        st.link_button("Mehr Infos",
                       "https://wandb.ai/mostafaibrahim17/ml-articles/reports/Extractive-Question-Answering-With-HuggingFace-Using-PyTorch-and-W-B--Vmlldzo0MzMwOTY5#what-is-bert")

        st.markdown('</div></div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="wix-col">', unsafe_allow_html=True)
    st.markdown('<div class="wix-title">DATENSATZ</div>', unsafe_allow_html=True)
    st.markdown('<div class="wix-text">GermanQuAD: annotierte Frage-Antwort-Paare mit Kontext, Frage und markierter Antwortspanne. Dient als Benchmark für deutsche QA-Modelle.</div>', unsafe_allow_html=True)
    st.markdown('<div class="wix-btn-wrap">', unsafe_allow_html=True)
    st.link_button("Mehr Infos","https://huggingface.co/datasets/deepset/germanquad")

    st.markdown('</div></div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="wix-col">', unsafe_allow_html=True)
    st.markdown('<div class="wix-title">PRINZIP</div>', unsafe_allow_html=True)
    st.markdown('<div class="wix-text">Das Modell schätzt pro Token die Wahrscheinlichkeit für Start/Ende der Antwort. Postprocessing (n-best) wählt daraus die plausibelste Spanne.</div>', unsafe_allow_html=True)
    st.markdown('<div class="wix-btn-wrap">', unsafe_allow_html=True)
    if st.button("Mehr Infos", key="more_principle"):
        st.switch_page("Pages/3_Mehr_Infos.py")
    st.markdown('</div></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# RENDER: INFO SEKTION
# -----------------------------
# -----------------------------
# RENDER: INFO SEKTION
# -----------------------------
st.markdown(f"""
<div class="info-shell">
  <div class="info-section">
    <div class="info-content">
      <div class="info-title">INFO</div>
      <div class="info-text">
        Das System basiert auf den Konzepten, die in der bahnbrechenden Arbeit <strong>„Attention is All You Need“</strong> von Vaswani et al. (2017) 
        beschrieben sind. Der Transformer-Ansatz mit Self-Attention ermöglicht es, lange Kontextbeziehungen effizient zu modellieren und ist somit das 
        Rückgrat moderner NLP-Modelle.
        Für die Fragen-Antwort-Systeme (QA) wurde das <strong>BERT-Modell (Bidirectional Encoder Representations from Transformers)</strong> von Devlin et al. (2018) verwendet, 
        das speziell für die span-basierte QA-Vorhersage entwickelt wurde. 
        In diesem Modell wird der relevante Text als Start- und End-Span im Kontext identifiziert, was zu präzisen Antwortbereichen führt.
      </div>
    </div>
""", unsafe_allow_html=True)

# Spalten für die Zentrierung der Buttons
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    c_left, c_right = st.columns(2)
    with c_left:
        # Link zum Attention Paper (NIPS)
        st.link_button(
            "Transformer Paper",
            "https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html"
        )
    with c_right:
        # Link zum BERT Paper (Google Research)
        st.link_button(
            "BERT Paper",
            "https://research.google/pubs/bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding/"
        )



st.markdown('</div></div>', unsafe_allow_html=True)

# -----------------------------
# CTA UNTEN
# -----------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;'><h3>Direkt zur Demo</h3>", unsafe_allow_html=True)
st.write("<div style='text-align:center;'>In der Demo kannst du fünf zufällige Fragen laden, Parameter verändern und Gold/Base/Post vergleichen.</div>", unsafe_allow_html=True)
cc1, cc2, cc3 = st.columns([1, 1, 1])
with cc2:
    if st.button("Zur Demo →", use_container_width=False):
        st.switch_page("Pages/1_Fragen_generieren.py")