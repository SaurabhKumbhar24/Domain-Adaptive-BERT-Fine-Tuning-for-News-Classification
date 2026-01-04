# Domain-Adaptive BERT Fine-Tuning for News Classification

## 🔍 Overview
This project demonstrates how **Domain-Adaptive Pretraining (DAPT)** improves downstream NLP performance by continuing **Masked Language Modeling (MLM)** on unlabeled, in-domain text before supervised fine-tuning.

Using the **AG News** dataset, we show that lightweight continued pretraining of BERT on news articles reduces domain mismatch and leads to measurable improvements in classification accuracy **without additional labeled data**.

---

## 🧠 Motivation
Pretrained language models like BERT are trained on large, general-purpose corpora. However, when applied to domain-specific tasks (e.g., news, medical, legal text), they often suffer from **distribution shift**.

Instead of collecting more labeled data, this project leverages **unlabeled in-domain text** to adapt the model’s internal representations efficiently.

---

## 📂 Dataset
- **AG News**
- 4-class topic classification:
  - World
  - Sports
  - Business
  - Sci/Tech
- ~120K training samples
- ~7.6K test samples

### Dataset Usage
- **Unlabeled text** → Domain-Adaptive Pretraining (MLM)
- **Labeled text** → Supervised fine-tuning

This separation ensures **no supervision leakage**.

---

## ⚙️ Methodology

### Phase 1: Data Preparation
- Loaded AG News using HuggingFace Datasets
- Extracted raw text to build an **unlabeled corpus** for DAPT
- Labels were ignored during pretraining

### Phase 2: Tokenization & MLM Formatting
- Used the original `bert-base-uncased` tokenizer
- Truncated sequences to 128 tokens for efficiency
- Applied **dynamic masking** using `DataCollatorForLanguageModeling`

### Phase 3: Domain-Adaptive Pretraining (DAPT)
- Loaded `BertForMaskedLM`
- Continued pretraining for **1 epoch** on news text using MLM
- Used AdamW optimizer with weight decay
- Saved the adapted model for downstream tasks

### Phase 4: Supervised Fine-Tuning
Two models were fine-tuned under **identical conditions**:
1. **Baseline**: `bert-base-uncased`
2. **DAPT Model**: BERT after news-domain MLM pretraining

---

## 📊 Results

| Model | Accuracy | F1 |
|------|----------|----|
| BERT (baseline) | 94.22% | 94.22% |
| **BERT + DAPT** | **94.46%** | **94.47%** |

**Δ Accuracy: +0.24%**  
**Δ F1: +0.25%**

---

## 🔍 Key Takeaways
- Domain mismatch affects downstream performance even for strong pretrained models
- Unlabeled in-domain text is highly valuable
- Lightweight DAPT can yield meaningful improvements with minimal compute
- Proper experimental control is essential
