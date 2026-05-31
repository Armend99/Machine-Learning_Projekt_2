import streamlit as st

# ============================================================
# Seiteneinstellung
# ============================================================
st.set_page_config(page_title="Funktionsweise: Transformer & Attention", layout="wide")

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 40px;
        font-weight: 700;
        color: #0d2a3f;
        text-align: center;
        margin-bottom: 8px;
    }
    .sub-title {
        font-size: 16px;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: 600;
        color: #0d2a3f;
        margin-top: 24px;
        margin-bottom: 10px;
    }
    .info-box {
        background-color: #e8f5ee;
        padding: 18px;
        border-radius: 10px;
        border-left: 5px solid #2a7a4f;
        margin: 18px 0;
    }
    .warn-box {
        background-color: #fefce8;
        padding: 18px;
        border-radius: 10px;
        border-left: 5px solid #d97706;
        margin: 18px 0;
    }
    .blue-box {
        background-color: #e8f0fa;
        padding: 18px;
        border-radius: 10px;
        border-left: 5px solid #1a4a6e;
        margin: 18px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f4f8;
        border-radius: 8px 8px 0 0;
        padding: 10px 18px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0d2a3f;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Header
# ============================================================
st.markdown('<div class="main-header">Wie funktioniert das Modell?</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Der gesamte Ablauf von der Frage bis zur Antwort — Schritt für Schritt am Beispiel</div>', unsafe_allow_html=True)

st.markdown("""
<div class="blue-box">
<strong>Unser durchgehendes Beispiel:</strong><br>
Frage: <em>"Wer gründete Microsoft?"</em><br>
Kontext: <em>"Bill Gates gründete Microsoft im Jahr 1975."</em><br><br>
Das Modell soll die Antwort <strong>"Bill Gates"</strong> aus dem Kontext extrahieren — es generiert keinen neuen Text, sondern markiert die richtige Stelle.
</div>
""", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1 · Tokens",
    "2 · Embeddings",
    "3 · Sliding Window",
    "4 · Encoder",
    "5 · Attention",
    "6 · Logits & Score"
])

# ------------------------------------------------------------
# TAB 1 — TOKENS
# ------------------------------------------------------------
with tab1:
    st.markdown('<div class="sub-header">Was ist ein Token?</div>', unsafe_allow_html=True)
    st.write("""
    Das Modell kann nicht mit Wörtern rechnen. Der erste Schritt ist daher die **Tokenisierung** — 
    der Text wird in die kleinsten verarbeitbaren Einheiten zerlegt. Ein Token ist dabei nicht immer 
    ein ganzes Wort, manchmal nur ein Wortteil.
    """)

    st.code(
"\"Bundeskanzler\"  ->  [\"Bundes\", \"##kanzler\"]   (2 Tokens)\n"
"\"Microsoft\"      ->  [\"Microsoft\"]               (1 Token)\n"
"\"QA-System\"      ->  [\"QA\", \"-\", \"System\"]       (3 Tokens)",
language=None)

    st.write("Das `##` bedeutet: dieses Stück gehört zum vorherigen Token (kein Leerzeichen davor).")

    st.markdown('<div class="sub-header">Unser Beispiel tokenisiert</div>', unsafe_allow_html=True)
    st.write("Frage und Kontext werden zu einer Sequenz zusammengefügt, getrennt durch Sondertokens:")

    st.code(
"[CLS]  Wer  gründete  Microsoft  ?  [SEP]  Bill  Gates  gründete  Microsoft  im  Jahr  1975  .  [SEP]\n"
"       |___________ Frage ___________|      |________________ Kontext ________________|",
language=None)

    st.markdown('<div class="sub-header">Rolle der Sondertokens</div>', unsafe_allow_html=True)
    st.markdown("""
    - **[CLS]** — steht immer am Anfang. Fasst den gesamten Input zusammen und dient als "Keine Antwort"-Signal, wenn der Kontext keine Antwort enthält.
    - **[SEP]** — trennt Frage und Kontext voneinander, und markiert das Ende.
    - **Kontext-Tokens** (Bill, Gates, ...) — nur hier wird die Antwort gesucht. Die Frage-Tokens kommen nie als Antwort infrage.
    """)

    st.markdown("""
    <div class="info-box">
    <strong>Wichtig:</strong> Jeder Token bekommt eine eindeutige ID (eine Zahl). Diese IDs sind die eigentliche Eingabe für das Modell.
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------
# TAB 2 — EMBEDDINGS
# ------------------------------------------------------------
with tab2:
    st.markdown('<div class="sub-header">Von der Token-ID zum Vektor</div>', unsafe_allow_html=True)
    st.write("""
    Eine Token-ID alleine trägt keine Bedeutung — die Zahl 1234 sagt nichts über das Wort "Wer" aus. 
    Deshalb wird jeder Token in einen **Embedding-Vektor** umgewandelt: eine Liste aus **768 Zahlen**, 
    die die Bedeutung des Tokens darstellt.
    """)

    st.code(
"Token \"Wer\"  ->  ID 1234  ->  T = [0.4, -0.2, 0.7, 0.1, ..., 0.3]\n"
"                              |___________ 768 Zahlen ___________|",
language=None)

    st.markdown('<div class="sub-header">Woher kommt der Vektor?</div>', unsafe_allow_html=True)
    st.write("""
    Aus einer **Embedding-Tabelle**, die im **Pre-Training** gelernt wurde. Das Modell hat dabei Millionen 
    deutsche Sätze gelesen. Das Besondere: Wörter mit ähnlicher Bedeutung bekommen ähnliche Vektoren.
    """)

    st.code(
"T(\"Bill\")    = [0.80, -0.31, 0.74, ...]\n"
"T(\"Gates\")   = [0.79, -0.28, 0.71, ...]\n"
"T(\"Merkel\")  = [0.81, -0.33, 0.73, ...]   <- alle ähnlich -> alle Personen\n"
"\n"
"T(\"1975\")    = [0.12,  0.91, -0.44, ...]  <- ganz anders -> Jahreszahl",
language=None)

    st.markdown("""
    <div class="info-box">
    <strong>Kernidee:</strong> Die Bedeutung eines Wortes ist seine Position im Zahlenraum. "Bill" und "Gates" 
    liegen nah beieinander, "1975" weit entfernt. Diese gelernte Bedeutung ist die Grundlage dafür, dass die 
    Attention später sinnvolle Verbindungen findet.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sub-header">Pre-Training vs. Fine-Tuning</div>', unsafe_allow_html=True)
    st.write("""
    Es ist wichtig, zwei Trainingsphasen zu unterscheiden:
    """)
    st.markdown("""
    - **Pre-Training (von deepset):** Das Modell lernt die deutsche Sprache allgemein — keine Fragen, keine Antworten. Ergebnis: das Embedding und die Attention-Gewichte.
    - **Fine-Tuning (dieses Projekt):** Das Modell lernt speziell Question Answering auf dem GermanQuAD-Datensatz — also wo im Kontext die Antwort steht.
    """)

# ------------------------------------------------------------
# TAB 3 — ENCODER
# ------------------------------------------------------------
with tab4:
    st.markdown('<div class="sub-header">Das Problem: Text von links nach rechts lesen</div>', unsafe_allow_html=True)
    st.write("""
    Stell dir vor, du liest einen Satz Token für Token von links nach rechts. Bei mehrdeutigen Wörtern 
    weißt du die Bedeutung erst spät:
    """)

    st.code(
"\"Die Bank am Fluss ist schön.\"\n"
"\n"
"Token \"Bank\"  ->  Geldinstitut? Sitzbank?  (noch unklar)\n"
"Token \"Fluss\" ->  jetzt klar: Sitzbank!    (zu spät)",
language=None)

    st.markdown('<div class="sub-header">Die Lösung: Bidirektionaler Encoder</div>', unsafe_allow_html=True)
    st.write("""
    Der Encoder von BERT/ELECTRA liest den Text **gleichzeitig von beiden Seiten**. Jeder Token sieht 
    alle anderen Tokens auf einmal — links und rechts.
    """)

    st.code(
"<-- \"Die Bank am Fluss ist schön.\" -->\n"
"\n"
"Token \"Bank\" sieht gleichzeitig \"Die\" (links) UND \"Fluss\" (rechts)\n"
"-> sofort klar: Sitzbank",
language=None)

    st.write("Das **B** in BERT steht für **B**idirectional — genau diese Eigenschaft.")

    st.markdown('<div class="sub-header">Der Encoder besteht aus 12 Schichten</div>', unsafe_allow_html=True)
    st.write("""
    Der Encoder ist kein einzelner Schritt, sondern 12 aufeinanderfolgende Schichten. Jede Schicht verfeinert 
    das Verständnis ein Stück weiter. Am Bill-Gates-Beispiel:
    """)

    st.code(
"Schicht 1:  \"Bill\" erkennt \"Gates\" als Nachbar\n"
"Schicht 4:  \"Bill Gates\" wird als ein Name erkannt\n"
"Schicht 8:  \"Wer\" verbindet sich mit \"Bill Gates\"\n"
"Schicht 12: \"Bill Gates\" = Antwort auf \"Wer\"",
language=None)

    st.markdown("""
    <div class="info-box">
    <strong>Wichtig:</strong> Der eigentliche Mechanismus, der in jeder dieser Schichten arbeitet, ist die 
    <strong>Attention</strong>. Die Bidirektionalität ist die Eigenschaft, dass die Attention in beide Richtungen 
    schauen darf. Das schauen wir uns im nächsten Tab im Detail an.
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------
# TAB 4 — ATTENTION
# ------------------------------------------------------------
with tab5:
    st.markdown('<div class="sub-header">Die Attention-Formel</div>', unsafe_allow_html=True)
    st.latex(r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \cdot V")

    st.write("Jeder Token erhält drei Rollen, die alle aus seinem Embedding-Vektor T berechnet werden:")
    st.markdown("""
    - **Q (Query)** — "Was suche ich?" — die Suchanfrage. Bei "Wer" = ich suche eine Person.
    - **K (Key)** — "Was biete ich an?" — das Angebot. Bei "Bill" = ich bin eine Person.
    - **V (Value)** — "Welche Information gebe ich weiter?" — der eigentliche Inhalt.
    """)

    st.code(
"Q = T · W_Q   (Suchanfrage)\n"
"K = T · W_K   (Angebot)\n"
"V = T · W_V   (Information)\n"
"\n"
"W_Q, W_K, W_V wurden im Pre-Training gelernt",
language=None)

    st.markdown('<div class="sub-header">Schritt 1: Ähnlichkeit berechnen (Q · K)</div>', unsafe_allow_html=True)
    st.write("""
    Das Skalarprodukt misst, wie ähnlich die Suchanfrage Q zu jedem Angebot K ist. Je ähnlicher die Vektoren, 
    desto höher die Zahl:
    """)
    st.code(
"Q(\"Wer\") · K(\"Bill\")      = 8.4   <- hoch\n"
"Q(\"Wer\") · K(\"Gates\")     = 7.9   <- hoch\n"
"Q(\"Wer\") · K(\"gründete\")  = 3.1   <- mittel\n"
"Q(\"Wer\") · K(\"Microsoft\") = 2.8   <- niedrig\n"
"Q(\"Wer\") · K(\"1975\")      = 1.2   <- sehr niedrig",
language=None)

    st.markdown('<div class="sub-header">Schritt 2: Skalierung (/ √d_k)</div>', unsafe_allow_html=True)
    st.write("""
    Die Werte werden durch √768 ≈ 27,7 geteilt. **Warum?** Bei 768 Dimensionen können die Skalarprodukte 
    sehr groß werden. Zu große Werte führen dazu, dass die Softmax fast die gesamte Aufmerksamkeit auf einen 
    einzigen Token legt und alle anderen ignoriert.
    """)

    st.markdown('<div class="sub-header">Schritt 3: Softmax</div>', unsafe_allow_html=True)
    st.latex(r"\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}")
    st.write("""
    Die Softmax wandelt die Werte in Wahrscheinlichkeiten um. Jeder Wert liegt zwischen 0 und 1, 
    die Summe ergibt genau 1,0:
    """)
    st.code(
"\"Bill\"      -> 38%\n"
"\"Gates\"     -> 34%\n"
"\"gründete\"  -> 13%\n"
"\"Microsoft\" -> 11%\n"
"\"1975\"      ->  4%\n"
"-----------------------\n"
"Summe       -> 100%",
language=None)

    st.markdown('<div class="sub-header">Schritt 4: Gewichtete Summe (· V)</div>', unsafe_allow_html=True)
    st.write("""
    Zuletzt werden die Gewichte mit den Value-Vektoren multipliziert und aufsummiert. Das Ergebnis ist 
    ein neuer, angereicherter Vektor für "Wer":
    """)
    st.code(
"neues T(\"Wer\") =  0.38 × V(\"Bill\")\n"
"               + 0.34 × V(\"Gates\")\n"
"               + 0.13 × V(\"gründete\")\n"
"               + 0.11 × V(\"Microsoft\")\n"
"               + 0.04 × V(\"1975\")",
language=None)

    st.markdown("""
    <div class="info-box">
    <strong>Was bedeutet das?</strong> Der neue Vektor von "Wer" ist eine gewichtete Mischung aller Tokens — 
    zu 72% aus "Bill" und "Gates". "Wer" hat damit das Wissen über "Bill Gates" in sich aufgenommen. 
    Genau das meint man mit Kontextverständnis.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="warn-box">
    <strong>Key vs. Value:</strong> Der Key (K) entscheidet, <em>ob</em> ein Token relevant ist. Der Value (V) 
    bestimmt, <em>welche Information</em> dann weitergegeben wird. Analogie Bibliothek: der Key ist der Titel auf 
    dem Buchrücken (damit findest du das Buch), der Value ist der Inhalt (das nimmst du tatsächlich mit).
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------
# TAB 5 — LOGITS & SCORE
# ------------------------------------------------------------
with tab6:
    st.markdown('<div class="sub-header">Vom Verständnis zur Antwortposition</div>', unsafe_allow_html=True)
    st.write("""
    Nach der Attention trägt jeder Token-Vektor T<sub>i</sub> das vollständige Kontextwissen. Jetzt bestimmt 
    das Modell, wo die Antwort beginnt und endet. Für jeden Token werden zwei Werte (Logits) berechnet:
    """, unsafe_allow_html=True)

    st.latex(r"S_i = T_i \cdot W_{start} \qquad E_i = T_i \cdot W_{end}")

    st.markdown("""
    - **S<sub>i</sub>** — wie wahrscheinlich ist dieser Token der **Start** der Antwort?
    - **E<sub>i</sub>** — wie wahrscheinlich ist dieser Token das **Ende** der Antwort?
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="warn-box">
    <strong>Pre-Training vs. Fine-Tuning:</strong> Das Sprachverständnis (Embedding, Attention) kommt aus dem 
    Pre-Training von deepset. Die Fähigkeit, Antworten zu markieren, wird im <strong>Fine-Tuning</strong> auf 
    GermanQuAD gelernt. Deshalb reichen vergleichsweise wenige Trainingsdaten.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sub-header">Logits am Beispiel</div>', unsafe_allow_html=True)
    data_logits = {
        "Token": ["Bill", "Gates", "gründete", "Microsoft", "1975"],
        "S (Start)": ["8.7", "2.1", "1.8", "1.4", "0.9"],
        "E (Ende)": ["1.2", "7.9", "2.3", "1.9", "1.1"]
    }
    st.table(data_logits)
    st.write("\"Bill\" hat den höchsten Start-Wert, \"Gates\" den höchsten End-Wert.")

    st.markdown('<div class="sub-header">Softmax und Score</div>', unsafe_allow_html=True)
    st.write("Die Logits werden per Softmax zu Wahrscheinlichkeiten. Dann wird für jedes Start-Ende-Paar ein Score berechnet:")

    st.latex(r"\text{Score}(i,j) = P(\text{start } i) \times P(\text{end } j), \quad j \ge i")

    st.code(
"Score(\"Bill\", \"Gates\")     = 72% × 68% = 49%   <- höchster!\n"
"Score(\"Bill\", \"gründete\")  = 72% × 14% = 10%\n"
"Score(\"Gates\", \"Gates\")    = 12% × 68% =  8%\n"
"Score(\"Gates\", \"Bill\")     = ungültig (Ende vor Start, j < i)",
language=None)

    st.markdown("""
    <div class="info-box">
    <strong>Ergebnis:</strong> Die Kombination "Bill" -> "Gates" hat den höchsten Score (49%). Das Modell 
    extrahiert den Text von Token "Bill" bis Token "Gates": <strong>"Bill Gates"</strong>.<br><br>
    Das ist <strong>Extractive QA</strong> — das Modell generiert keinen neuen Text, es markiert nur die 
    richtige Stelle im Kontext.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sub-header">Die Softmax — ein wiederkehrendes Werkzeug</div>', unsafe_allow_html=True)
    st.write("""
    Es ist dieselbe Softmax-Formel wie bei der Attention — nur der Input ist anders. Bei der Attention sind es 
    Ähnlichkeiten zwischen Tokens, hier sind es die Start- und End-Logits. Einmal verstanden, mehrfach angewendet.
    """)

# ------------------------------------------------------------
# TAB 6 — SLIDING WINDOW
# ------------------------------------------------------------
with tab3:
    st.markdown('<div class="sub-header">Das Problem bei langen Texten</div>', unsafe_allow_html=True)
    st.write("""
    Das Modell kann maximal **512 Tokens** auf einmal verarbeiten. Ein Wikipedia-Artikel ist oft viel länger:
    """)
    st.code(
"Wikipedia-Artikel:  1.200 Tokens\n"
"Modell-Limit:         512 Tokens\n"
"                       -> 700 Tokens würden abgeschnitten!\n"
"                          Die Antwort könnte darin liegen.",
language=None)

    st.markdown('<div class="sub-header">Die Lösung: Sliding Window</div>', unsafe_allow_html=True)
    st.write("""
    Der lange Text wird in überlappende Fenster (Chunks) zerlegt. Jeder Chunk wird separat durch das Modell 
    geschickt — der Chunk mit dem höchsten Antwort-Score gewinnt.
    """)
    st.code(
"Chunk 1: Token   1 -> 512\n"
"Chunk 2: Token 385 -> 896   <- Überlappung mit Chunk 1\n"
"Chunk 3: Token 769 -> 1200",
language=None)

    st.markdown('<div class="sub-header">Warum Überlappung? (doc_stride)</div>', unsafe_allow_html=True)
    st.write("""
    Der **doc_stride** legt fest, wie stark sich die Chunks überlappen. Ohne Überlappung kann eine Antwort 
    genau an einer Chunk-Grenze zerrissen werden:
    """)
    st.code(
"OHNE Überlappung:\n"
"  Chunk 1 endet:    \"...gründete Bill\"\n"
"  Chunk 2 beginnt:  \"Gates Microsoft...\"\n"
"  -> \"Bill Gates\" wird zerrissen!\n"
"\n"
"MIT Überlappung (doc_stride = 128):\n"
"  Chunk 1: \"...gründete Bill Gates Microsoft...\"\n"
"  Chunk 2: \"...gründete Bill Gates Microsoft...\"\n"
"  -> \"Bill Gates\" ist in beiden Chunks vollständig",
language=None)

    st.markdown("""
    <div class="info-box">
    <strong>Wichtig:</strong> Das Sliding Window betrifft nur den <strong>Kontext</strong>, nicht die Frage. 
    Im Code sorgt <code>truncation="only_second"</code> dafür, dass nur der Kontext gekürzt wird — die Frage 
    bleibt immer vollständig erhalten.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="warn-box">
    <strong>Trade-off:</strong> Kleiner doc_stride = mehr Überlappung = sicherer, aber mehr Chunks (langsamer). 
    Großer doc_stride = weniger Überlappung = schneller, aber Antworten an Grenzen können verloren gehen.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# Zurück-Button
# ============================================================
st.markdown("---")
if st.button("Zurück zur Startseite"):
    st.switch_page("app.py")

