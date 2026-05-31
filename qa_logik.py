# qa_logik.py
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

def load_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device

def _prepare(tokenizer, question, context, max_seq_length=512, doc_stride=128):
    tok = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    # offsets nur für Kontext behalten
    for i in range(len(tok["offset_mapping"])):
        seq_ids = tok.sequence_ids(i)
        tok["offset_mapping"][i] = [
            o if seq_ids[j] == 1 else None
            for j, o in enumerate(tok["offset_mapping"][i])
        ]
    return tok

@torch.no_grad()
def predict_gold_base_post(
    tokenizer, model, device,
    question, context, gold_text,
    n_best_size=30, max_answer_length=50,
    max_seq_length=512, doc_stride=128
):
    feats = _prepare(tokenizer, question, context, max_seq_length, doc_stride)
    input_ids = torch.tensor(feats["input_ids"]).to(device)
    attention = torch.tensor(feats["attention_mask"]).to(device)

    out = model(input_ids=input_ids, attention_mask=attention)
    s_logits = out.start_logits.cpu().numpy()
    e_logits = out.end_logits.cpu().numpy()

    # BASE: Top-1 Span (argmax)
    base = {"score": -1e9, "text": ""}
    for c in range(s_logits.shape[0]):
        offs = feats["offset_mapping"][c]
        s = int(np.argmax(s_logits[c]))
        e = int(np.argmax(e_logits[c]))
        if e < s or (e - s + 1) > max_answer_length:
            continue
        if offs[s] is None or offs[e] is None:
            continue
        a, b = offs[s][0], offs[e][1]
        sc = float(s_logits[c][s] + e_logits[c][e])
        if sc > base["score"]:
            base = {"score": sc, "text": context[a:b]}

    # POST: n-best Span
    post = {"score": -1e9, "text": ""}
    for c in range(s_logits.shape[0]):
        offs = feats["offset_mapping"][c]
        ss = np.argsort(s_logits[c])[-n_best_size:][::-1]
        ee = np.argsort(e_logits[c])[-n_best_size:][::-1]
        for s in ss:
            for e in ee:
                if e < s or (e - s + 1) > max_answer_length:
                    continue
                if offs[int(s)] is None or offs[int(e)] is None:
                    continue
                a, b = offs[int(s)][0], offs[int(e)][1]
                sc = float(s_logits[c][int(s)] + e_logits[c][int(e)])
                if sc > post["score"]:
                    post = {"score": sc, "text": context[a:b]}

    return {
        "gold": gold_text,
        "base": base["text"],
        "post": post["text"],
        "base_score": base["score"],
        "post_score": post["score"],
    }