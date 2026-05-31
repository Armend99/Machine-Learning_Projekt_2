import subprocess
import time
import qrcode
from pyngrok import ngrok

# Streamlit App im Hintergrund starten
process = subprocess.Popen(
    ["streamlit", "run", "app.py", "--server.headless", "true"],
    cwd="C:\\Users\\armen\\Neuronale Netzwerke"
)

# Länger warten bis Streamlit vollständig hochgefahren ist
print("⏳ Warte auf Streamlit...")
time.sleep(10)  # ← 10 statt 3 Sekunden

# Öffentliche URL via ngrok erzeugen
public_url = ngrok.connect(8501)
url_str = public_url.public_url
print("✅ App läuft!")
print("🌍 Öffentliche URL:", url_str)

# QR Code generieren und speichern
qr = qrcode.make(url_str)
qr.save("qr_code.png")
print("📱 QR Code gespeichert als qr_code.png")

# Prozess am Laufen halten
input("Drücke Enter zum Beenden...")
process.terminate()