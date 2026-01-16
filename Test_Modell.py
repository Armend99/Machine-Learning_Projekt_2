
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

MODEL_DIR = "Model/qa_gelectra"  # oder dein aktueller Pfad

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_DIR)

print(model.config._name_or_path)