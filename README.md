# Aspect-Based Sentiment Analysis (ABSA)

A two-stage NLP pipeline that identifies **what** aspects are discussed in a review and determines **how** the author feels about each one — built on fine-tuned `bert-base-uncased` models and published to the Hugging Face Hub.

---

## How it works

```
Review text
    │
    ▼
┌─────────────────────────────────┐
│   ASE Model (Token Classification)  │   → extracts aspect spans
│   UseCondomsKid/ase-model       │     e.g. "battery life", "screen"
└─────────────────────────────────┘
    │  (text, aspect) pairs
    ▼
┌─────────────────────────────────┐
│  ABSA Model (Seq. Classification)   │   → classifies sentiment per aspect
│   UseCondomsKid/absa-model      │     positive / negative / neutral
└─────────────────────────────────┘
    │
    ▼
Structured JSON output with per-aspect sentiments + aggregated summary
```

**Stage 1 — Aspect Span Extraction (ASE):** A token classification model tags each word with `B-ASP`, `I-ASP`, or `O` (BIO scheme). Contiguous aspect tokens are merged into aspect terms.

**Stage 2 — Aspect Sentiment Classification (ABSC):** The review text and each extracted aspect term are fed together as a sentence pair into a sequence classifier that outputs `positive`, `negative`, or `neutral`.

---

## Repository structure

```
.
├── datasets/
│   ├── mams/                   # MAMS dataset (train.xml, val.xml, test.xml)
│   └── semeval2014/            # SemEval-2014 Task 4 (laptop.xml, restaurants.xml)
├── prepared_data/
│   ├── ase_dataset/            # Preprocessed token-classification dataset
│   └── absa_dataset/           # Preprocessed sequence-classification dataset
├── ase_model/                  # ASE model checkpoints + best/
├── absa_model/                 # ABSA model checkpoints + best/
├── ABSA.ipynb                  # Full training pipeline (data prep → train → push)
├── inference.py                # Ready-to-use inference module (loads from HF Hub)
└── README.md
```

---

## Datasets

| Dataset | Split | Used for |
|---|---|---|
| [MAMS](https://github.com/siat-nlp/MAMS-for-ABSA) | train / val / test | Primary data for all splits |
| [SemEval-2014 Task 4](https://aclanthology.org/S14-2004/) | — (merged into train) | Extra training examples (laptops + restaurants) |

Samples with `conflict` polarity are excluded from the ABSA dataset. The final label set is `positive`, `negative`, `neutral`.

---

## Models

Both models are fine-tuned from `bert-base-uncased` and hosted on the Hugging Face Hub.

### ASE — Aspect Span Extraction
**[`UseCondomsKid/ase-model`](https://huggingface.co/UseCondomsKid/ase-model)**

- Architecture: `BertForTokenClassification`
- Labels: `O`, `B-ASP`, `I-ASP`
- Max sequence length: 128
- Evaluation metric: seqeval F1

| Hyperparameter | Value |
|---|---|
| Epochs | 5 |
| Learning rate | 3e-5 |
| Train batch size | 16 |
| Eval batch size | 32 |
| Warmup steps | 100 |
| Weight decay | 0.01 |
| Best checkpoint | highest eval F1 |

### ABSA — Sentiment Classification
**[`UseCondomsKid/absa-model`](https://huggingface.co/UseCondomsKid/absa-model)**

- Architecture: `BertForSequenceClassification`
- Labels: `positive` (0), `negative` (1), `neutral` (2)
- Input format: `[CLS] review text [SEP] aspect term [SEP]`
- Max sequence length: 128
- Evaluation metric: macro F1

| Hyperparameter | Value |
|---|---|
| Epochs | 5 |
| Learning rate | 2e-5 |
| Train batch size | 16 |
| Eval batch size | 32 |
| Warmup steps | 100 |
| Weight decay | 0.01 |
| Best checkpoint | highest eval macro F1 |

---

## Setup

### Requirements

- Python 3.10+
- PyTorch (with CUDA for GPU inference)

### Install dependencies

```bash
pip install transformers torch numpy
```

### Clone the repo

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

---

## Inference

`inference.py` loads both models directly from the Hugging Face Hub — no local training required.

```python
from inference import analyze_reviews

reviews = [
    "The battery life is amazing but the screen is terrible.",
    "Great camera and fast delivery, packaging was a bit disappointing.",
    "Solid build quality, the price is fair for what you get.",
]

output = analyze_reviews(reviews)
```

### Output format

```json
{
  "overall_score": 0.6842,
  "overall_label": "positive",
  "summary": {
    "positive": 4,
    "negative": 3,
    "neutral": 1,
    "total_aspects": 8
  },
  "reviews": [
    {
      "text": "The battery life is amazing but the screen is terrible.",
      "aspects": [
        { "term": "battery life", "label": "positive", "score": 0.9321 },
        { "term": "screen",       "label": "negative", "score": 0.1204 }
      ],
      "review_score": 0.5263
    }
  ],
  "aspect_summary": {
    "battery life": { "positive": 1, "negative": 0, "neutral": 0, "avg_score": 0.9321 },
    "screen":       { "positive": 0, "negative": 1, "neutral": 0, "avg_score": 0.1204 }
  }
}
```

### Individual functions

```python
from inference import extract_aspects, get_sentiment

# Stage 1: extract aspect terms from text
aspects = extract_aspects("Great camera and fast delivery, packaging was a bit disappointing.")
# → ["camera", "delivery", "packaging"]

# Stage 2: classify sentiment for a specific aspect
result = get_sentiment("Great camera and fast delivery.", "camera")
# → {"label": "positive", "score": 0.9145}
```

---

## Training

To reproduce training from scratch, open `ABSA.ipynb` and run all cells in order.

**Step 1 — Prepare data**

Place the raw datasets at the expected paths:

```
datasets/
├── mams/
│   ├── train.xml
│   ├── val.xml
│   └── test.xml
└── semeval2014/
    ├── laptop.xml
    └── restaurants.xml
```

**Step 2 — Run the notebook**

The notebook will:
1. Parse and merge the XML datasets
2. Tokenize and build BIO labels for ASE
3. Build sentence-pair examples for ABSA
4. Train the ASE model and save the best checkpoint to `./ase_model/best`
5. Train the ABSA model and save the best checkpoint to `./absa_model/best`
6. Push both models and tokenizers to the Hugging Face Hub

**Step 3 — Push to Hub** *(optional)*

You will be prompted to log in with `huggingface_hub.login()`. Update the Hub repo names in the last notebook cell before pushing.

---

## License

This project is released under the MIT License.
