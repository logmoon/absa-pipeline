import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
)

ASE_PATH  = "UseCondomsKid/ase-model"
ABSA_PATH = "UseCondomsKid/absa-model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[inference] Loading models on {device}...")

ase_tokenizer = AutoTokenizer.from_pretrained(ASE_PATH)
ase_model     = AutoModelForTokenClassification.from_pretrained(ASE_PATH).to(device).eval()

absa_tokenizer = AutoTokenizer.from_pretrained(ABSA_PATH)
absa_model     = AutoModelForSequenceClassification.from_pretrained(ABSA_PATH).to(device).eval()

print("[inference] Models ready.")


def extract_aspects(text: str) -> list[str]:
    inputs   = ase_tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    word_ids = ase_tokenizer(text, truncation=True, max_length=128).word_ids()

    with torch.no_grad():
        logits = ase_model(**inputs).logits

    preds    = torch.argmax(logits, dim=-1)[0].tolist()
    tokens   = ase_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    id2label = ase_model.config.id2label
    aspects  = []
    current  = []
    prev_word_id = None

    def flush():
        if current:
            term = ""
            for t in current:
                term += t[2:] if t.startswith("##") else ("" if not term else " ") + t
            aspects.append(term.strip())
            current.clear()

    for token, pred, word_id in zip(tokens, preds, word_ids):
        if word_id is None:
            flush()
            prev_word_id = None
            continue
        if word_id != prev_word_id:
            label = id2label[pred]
            if label == "B-ASP":
                flush()
                current.append(token)
            elif label == "I-ASP" and current:
                current.append(token)
            else:
                flush()
        prev_word_id = word_id

    flush()
    return [a for a in aspects if a]


def get_sentiment(text: str, aspect: str) -> dict:
    inputs = absa_tokenizer(
        text, aspect, return_tensors="pt", truncation=True, max_length=128
    ).to(device)
    with torch.no_grad():
        logits = absa_model(**inputs).logits
    probs     = torch.softmax(logits, dim=-1)[0].tolist()
    label_idx = int(np.argmax(probs))
    return {
        "label": absa_model.config.id2label[label_idx],
        "score": round(probs[0], 4),
    }


def analyze_reviews(reviews: list[str]) -> dict:
    all_reviews      = []
    aspect_summary   = {}
    sentiment_scores = []

    for text in reviews:
        aspects        = extract_aspects(text)
        aspect_results = []

        for aspect in aspects:
            sentiment = get_sentiment(text, aspect)
            aspect_results.append({"term": aspect, **sentiment})

            if aspect not in aspect_summary:
                aspect_summary[aspect] = {"positive": 0, "negative": 0, "neutral": 0, "scores": []}
            aspect_summary[aspect][sentiment["label"]] += 1
            aspect_summary[aspect]["scores"].append(sentiment["score"])

        review_score = (
            round(float(np.mean([a["score"] for a in aspect_results])), 4)
            if aspect_results else None
        )
        all_reviews.append({"text": text, "aspects": aspect_results, "review_score": review_score})
        if review_score is not None:
            sentiment_scores.append(review_score)

    for asp in aspect_summary:
        scores = aspect_summary[asp].pop("scores")
        aspect_summary[asp]["avg_score"] = round(float(np.mean(scores)), 4)

    overall_score = round(float(np.mean(sentiment_scores)), 4) if sentiment_scores else None
    label_counts  = {"positive": 0, "negative": 0, "neutral": 0}
    for r in all_reviews:
        for a in r["aspects"]:
            label_counts[a["label"]] += 1

    return {
        "overall_score" : overall_score,
        "overall_label" : max(label_counts, key=label_counts.get),
        "summary"       : {**label_counts, "total_aspects": sum(label_counts.values())},
        "reviews"       : all_reviews,
        "aspect_summary": aspect_summary,
    }
