
```markdown
# 🗣️ Speaker Turn Dynamics and Self-Attention for Dialogue Act Classification

This repository contains the code and experiments for the project **"Speaker Turn Dynamics and Self-Attention for Dialogue Act Classification"**, conducted as part of the **Text Analytics course at the University of Oklahoma**.

## 📚 Project Overview

This project revisits the task of Dialogue Act Classification (DAC) by proposing two key enhancements over a prior state-of-the-art model (He et al., 2021):

1. **Transformer Encoder:** We replace the BiGRU-based sequential modeling component with a Transformer encoder to capture long-range dependencies in multi-turn conversations.
2. **Speaker Turn Embedding:** We introduce refined speaker turn embeddings that distinguish between “speaker continued” and “speaker switched” utterances, allowing the model to learn turn-taking dynamics more effectively.

All other aspects of the original model—RoBERTa embeddings, loss function, chunking strategy, and evaluation pipeline—remain unchanged to ensure fair and interpretable comparisons.

---

## 📊 Datasets Used

Experiments were conducted on three benchmark datasets:
- **SwDA (Switchboard Dialogue Act Corpus)** – 43 classes
- **MRDA (ICSI Meeting Recorder Dialogue Act Corpus)** – 5 classes
- **DyDA (DailyDialog)** – 4 classes

Each dataset includes multi-turn dialogues annotated with fine-grained dialogue act labels.

---

## 🧠 Model Architecture

```

RoBERTa \[CLS] embeddings
│
+-------▼--------+
\| Turn-aware     | (0 = continued, 1 = switched, 2 = padding)
\| Speaker Embeds |
+-------▼--------+
│
Transformer Encoder (PyTorch)
│
Linear Classifier
│
Dialogue Act Prediction

````

- **Backbone:** `roberta-base` from Huggingface Transformers
- **Encoder:** Multi-head Transformer (replaces original BiGRU)
- **Speaker Turn Embedding:** Added to the sentence embeddings before encoding

---

## 🧪 Results Summary

| Dataset | Val Accuracy | Test Accuracy | Baseline (He et al., 2021) | Improvement |
|---------|--------------|---------------|-----------------------------|-------------|
| SwDA    | 77.8%        | 76.6%         | ~74.5%                      | ✅ +2.1%    |
| MRDA    | 88.8%        | 90.3%         | ~87.0%                      | ✅ +3.3%    |
| DyDA    | 80.6%        | 83.6%         | ~79.2%                      | ✅ +4.4%    |

---

## 🚀 How to Run

### Setup
```bash
git clone https://github.com/ferialnajiantabriz/dialogue-act-transformer.git
cd dialogue-act-transformer
pip install -r requirements.txt
````

### Train on a Dataset (e.g., SwDA)

```bash
python engine.py --corpus swda --mode train --nclass 43 \
                 --batch_size 8 --chunk_size 32 \
                 --nlayer 1 --dropout 0.5 \
                 --speaker_info emb_cls \
                 --nfinetune 2 --lr 1e-4
```

### Evaluate

```bash
python engine.py --corpus swda --mode inference
```

---

## 📁 Repository Structure

```
├── datasets.py          # Data preprocessing and turn-aware embedding logic
├── models.py            # Transformer-based model with speaker turn embedding
├── engine.py            # Training, evaluation, and inference logic
├── data/                # CSV files per dataset (train/val/test)
└── report.pdf           # Final LaTeX-formatted report (10 pages)
```

---

## 🧾 Citation & Acknowledgment

This project is based on the original code and model proposed by:

**He, R., Deng, Y., Yu, C., & Zhang, Z. (2021).**
*Speaker Turn Modeling for Dialogue Act Classification.*
[Paper PDF](https://www.researchgate.net/publication/357385651_Speaker_Turn_Modeling_for_Dialogue_Act_Classification)

We thank Professor **Dr. Jie Cao** for feedback and guidance during the project.

---

## 📌 License

This repository is for academic and educational use only. All rights to the original datasets and pre-trained models belong to their respective authors.
