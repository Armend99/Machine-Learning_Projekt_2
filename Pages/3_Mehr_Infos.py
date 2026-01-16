import streamlit as st

# Seiteneinstellung
st.set_page_config(page_title="Prinzip: Token-Klassifizierung", layout="wide")

# Custom CSS für ein sauberes Design
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: 600;
        color: #0d2a3f;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 28px;
        font-weight: 500;
        color: #0d2a3f;
        margin-top: 30px;
    }
    .info-box {
        background-color: #f0f4f7;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #0d2a3f;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Mathematisches Prinzip: Extractive QA</div>', unsafe_allow_html=True)

st.write("""
Das Ziel eines extraktiven Question-Answering-Systems (wie BERT oder RoBERTa) ist es nicht, neuen Text zu generieren. 
Stattdessen lernt das Modell, die präzise **Start-** und **Endposition** der Antwort innerhalb eines gegebenen Kontextes zu identifizieren.
""")

st.markdown('<div class="sub-header">1. Input: Die Token-Ebene</div>', unsafe_allow_html=True)
st.write("""
Bevor das Modell Berechnungen anstellt, wird der Text in **Tokens** (Wörter oder Teilwörter) zerlegt. 
Um Frage und Kontext für das Modell unterscheidbar zu machen, wird eine spezielle Struktur verwendet:
""")

st.code("[CLS] Frage [SEP] Kontext [SEP]", language=None)

st.markdown('<div class="sub-header">2. Die mathematische Vorhersage</div>', unsafe_allow_html=True)
st.write("""
Für jedes Token $i$ im Kontext berechnet das Modell zwei Werte (Logits): $S_i$ (für den Start) und $E_i$ (für das Ende).
Diese Logits entstehen durch das Skalarprodukt des Token-Vektors $T_i$ mit gelernten Gewichtsvektoren:
""")

st.latex(r"S_i = T_i \cdot W_{start}")
st.latex(r"E_i = T_i \cdot W_{end}")

st.write("Um aus diesen Werten echte Wahrscheinlichkeiten zu machen, wird die **Softmax-Funktion** angewendet:")

col1, col2 = st.columns(2)
with col1:
    st.latex(r"P(\text{start } i) = \frac{e^{S_i}}{\sum_j e^{S_j}}")
with col2:
    st.latex(r"P(\text{end } i) = \frac{e^{E_i}}{\sum_j e^{E_j}}")

st.markdown('<div class="sub-header">3. Bestimmung der besten Antwortspanne</div>', unsafe_allow_html=True)
st.write("""
Das System sucht nach dem Paar aus Startindex $i$ und Endindex $j$, das das Produkt der Wahrscheinlichkeiten maximiert. 
Dabei gilt die Bedingung $j \ge i$ (das Ende kann nicht vor dem Anfang liegen).
""")

st.latex(r"\text{Score}_{(i,j)} = P(\text{start } i) \times P(\text{end } j)")

st.markdown('<div class="sub-header">4. Beispielhafte Wahrscheinlichkeitsverteilung</div>', unsafe_allow_html=True)
st.write("Kontext: *'Berlin ist die Hauptstadt von Deutschland.'*")

# Beispiel-Tabelle
data = {
    "Token (i)": ["Berlin", "ist", "die", "Hauptstadt", "von", "Deutschland", "."],
    "P (Start)": ["85.2 %", "2.1 %", "0.5 %", "1.2 %", "0.1 %", "0.1 %", "0.0 %"],
    "P (Ende)": ["0.2 %", "0.3 %", "0.1 %", "1.5 %", "0.4 %", "92.8 %", "4.7 %"]
}
st.table(data)

st.markdown("""
<div class="info-box">
    <strong>Ergebnis:</strong> In diesem Fall würde das Modell "Berlin" als Start (85.2%) und "Deutschland" als Ende (92.8%) wählen. 
    Die extrahierte Antwortspanne lautet somit: <strong>"Berlin ist die Hauptstadt von Deutschland"</strong>.
</div>
""", unsafe_allow_html=True)

if st.button("Zurück zur Startseite"):
    st.switch_page("app.py")