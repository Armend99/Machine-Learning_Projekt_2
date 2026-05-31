# Machine Learning Projekt 2 — Extractive Question Answering (Deutsch)

Ein extraktives Question-Answering-System auf Basis von **gelectra-base-germanquad** (Transformer / Attention), fine-getunt auf dem **GermanQuAD**-Datensatz. Das Modell findet zu einer Frage die passende Antwort innerhalb eines gegebenen Kontextes.

## Funktionsweise

Das Modell extrahiert Antworten, indem es Start- und End-Position der Antwort im Kontext bestimmt:

1. **Tokenisierung** — Frage und Kontext werden in Tokens zerlegt
2. **Embedding** — jeder Token wird zu einem Vektor mit Bedeutung
3. **Encoder & Attention** — der bidirektionale Encoder verknüpft alle Tokens miteinander
4. **Logits & Score** — das Modell bestimmt die wahrscheinlichste Antwortspanne

## Ergebnisse

| Metrik | Wert (Testset, n=2.204) |
|--------|--------------------------|
| Exact Match | 63,75 % |
| F1-Score | 79,54 % |

## Projektstruktur

```
app.py                       Hauptseite (Streamlit)
Pages/
  1_Fragen_generieren.py     Interaktive Demo
  3_Mehr_Infos.py            Funktionsweise erklärt
qa_logik.py                  Modell-Logik
Model/qa_gelectra_v2/        Das fine-getunte Modell
start.py                     Demo-Start mit QR-Code (ngrok)
```

## ⚠️ Hinweis zum Modell

Die Modell-Gewichtsdatei **`model.safetensors`** (ca. 417 MB) ist **nicht** in diesem Repository enthalten, da sie das GitHub-Limit von 100 MB überschreitet.

Alle übrigen Modelldateien (Konfiguration, Tokenizer, Vokabular) sind vorhanden — es fehlt nur die eigentliche Gewichtsdatei.

**Wenn du das vollständige Modell brauchst, um die Anwendung auszuführen, sprich mich einfach an — ich stelle dir die Datei gerne zur Verfügung.**

## Anwendung starten

```bash
pip install -r requirements.txt
streamlit run app.py
```

(Setzt voraus, dass `Model/qa_gelectra_v2/model.safetensors` vorhanden ist.)
