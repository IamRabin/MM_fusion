# Multimodal Framework for Non-Invasive Prediction of Dementia Biomarker p-tau217

This repository contains the implementation of a **multimodal deep learning framework** for the **non-invasive prediction of the Alzheimer’s disease biomarker p-tau217** using **EEG signals and tabular clinical features**.

The framework combines a **pretrained EEG foundation model (EEG-VJEPA)** with **clinical and genetic metadata** using **FiLM-based multimodal fusion** to estimate biomarker levels from non-invasive data.

The goal of this project is to explore scalable machine learning methods for **biomarker prediction without lumbar puncture or PET imaging**, potentially enabling earlier and more accessible screening for neurodegenerative diseases.

---

# Overview

The system integrates two modalities:

1. **EEG signals**
2. **Tabular clinical/genetic features**

These modalities are encoded independently and fused using **Feature-wise Linear Modulation (FiLM)** before performing regression.

### Model Pipeline
EEG signals (.npy)
│
▼
AdaptiveBlock (shape alignment)
│
EEG-VJEPA pretrained encoder (Vision Transformer)
│
Token embeddings
│
Attention pooling
│
EEG embedding

Tabular features
│
MLP encoder
│
Tabular embedding

FiLM conditioning
│
Concatenation
│
Residual fusion MLP
│
Regression head
▼
Predicted p-tau217 level



# Key Components

## EEG Encoder

EEG representations are obtained using a **pretrained EEG-VJEPA encoder** based on Vision Transformers.

Processing steps:

1. EEG reshaped using `AdaptiveBlock`
2. Passed through pretrained encoder
3. Token embeddings pooled using **attention pooling**
4. Final EEG embedding extracted

---

## Tabular Encoder

Tabular features (demographic, genetic, and clinical metadata) are encoded using a **multi-layer perceptron (MLP)** with:

- Batch normalization  
- Dropout regularization  
- Non-linear activation functions  

---

## Multimodal Fusion

Fusion between EEG and tabular embeddings is performed using **Feature-wise Linear Modulation (FiLM)**:


---

## Regression Head

The final predictor consists of:

- Linear projection
- Residual MLP blocks
- Regression output layer

The model is trained using **Huber Loss**, which is robust to outliers.

---

# Dataset

The dataset contains:

- **EEG recordings** stored as `.npy` tensors
- **Tabular metadata** stored in a CSV file
- Target variable: **p-tau217 level**

Participant-level splitting is used to prevent **data leakage across subjects**.
train: 70%
validation: 15%
test: 15%

# Requirements

Python ≥ 3.9

Main dependencies:


torch
numpy
pandas
scikit-learn
matplotlib
seaborn
umap-learn
tqdm

# Running the Code
Train the model -->
python main.py
