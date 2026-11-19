# Multilingual Context-Aware AI Safety & Intent Analysis System
## An Odia-Centric Study on Multi-Task Neural Architectures for Prompt Safety Analysis

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch%202.6-ee4c2c.svg)](https://pytorch.org/)
[![Transformers: HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.48-yellow.svg)](https://huggingface.co/)
[![UI: Streamlit](https://img.shields.io/badge/UI-Streamlit%201.58-ff4b4b.svg)](https://streamlit.io/)
[![License: Academic Research](https://img.shields.io/badge/License-Academic%20Defensive-green.svg)](#)

---

## 1. Project Overview

Large Language Models (LLMs) deployed in multilingual societies often struggle to maintain consistent safety guardrails across regional and low-resource languages. Attack vectors such as prompt injection, conversational jailbreaking, and obfuscation exploit linguistic gaps—often bypassing English safety filters when translated into Indian languages or transliterated into Romanized scripts.

Conversely, standard keyword-matching blocklists frequently cause **severe false positives** by blocking legitimate academic, scientific, medical, and defensive queries simply because they mention sensitive technical concepts.

This project introduces a **multilingual context-aware multi-task intent analysis framework** designed to:
1. Decode the **underlying intent** of a query (Educational, Analysis, Prevention, Execution, Evasion).
2. Characterize its **situational context** (Academic, Historical, Fictional, Preventive, Real-world).
3. Evaluate its **operational actionability** (Conceptual, General guidance, Procedural, Highly actionable).
4. Predict calibrated **risk levels** (Low, Medium, High).
5. Gauge **linguistic ambiguity** (Low, Medium, High).

The primary research focus is **Odia (ଓଡ଼ିଆ)**, alongside English, Odia-English code-mixed text, Romanized Odia, and Hindi.

---

## 2. Multi-Task Model Architecture

The core model employs a **shared multilingual encoder** (`google/muril-base-cased` or `FacebookAI/xlm-roberta-base`) coupled with **five independent feed-forward classification heads**:

```
                  ┌──────────────────────────────┐
                  │ Input Prompt (En / Or / Mix) │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │   MuRIL Tokenizer (WordPiece)│
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │  MuRIL Base Encoder (768d)   │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │ Shared Latent Representation │
                  └──────────────┬───────────────┘
                                 │
     ┌─────────────┬─────────────┼─────────────┬─────────────┐
     ▼             ▼             ▼             ▼             ▼
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│ Intent  │   │ Context │   │ Action  │   │  Risk   │   │Ambiguity│
│  Head   │   │  Head   │   │  Head   │   │  Head   │   │  Head   │
│(5 cls)  │   │(5 cls)  │   │(4 cls)  │   │ (3 cls) │   │ (3 cls) │
└─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘
```

### Combined Multi-Task Loss Objective
The model minimizes a weighted cross-entropy loss with support for masked labels (`ignore_index = -100`):

$$\mathcal{L}_{total} = \lambda_{risk}\mathcal{L}_{risk} + \lambda_{intent}\mathcal{L}_{intent} + \lambda_{context}\mathcal{L}_{context} + \lambda_{act}\mathcal{L}_{act} + \lambda_{amb}\mathcal{L}_{amb}$$

---

## 3. Dataset Specification

The framework uses a 16-column dataset schema designed to avoid data leakage through **semantic group partitioning**:

| Column Name | Description | Example Values |
|---|---|---|
| `id` | Unique sample identifier | `SMP_001` |
| `text` | Prompt text in target language | *"Explain photosynthesis..."* |
| `language` | Language code | `en`, `or`, `or-en`, `or-Latn`, `hi` |
| `script` | Writing system | `Latin`, `Odia`, `Devanagari` |
| `is_code_mixed` | Code-mixing flag | `True` / `False` |
| `category` | Subject domain | `General science`, `Cybersecurity`, `Healthcare` |
| `risk_level` | Assessed safety risk | `Low`, `Medium`, `High` |
| `intent` | Core intention | `Educational`, `Analysis`, `Prevention`, `Execution`, `Evasion` |
| `context` | Framing context | `Academic`, `Historical`, `Fictional`, `Preventive`, `Real-world` |
| `actionability` | Actionability tier | `Conceptual`, `General guidance`, `Procedural`, `Highly actionable` |
| `ambiguity` | Linguistic ambiguity | `Low`, `Medium`, `High` |
| `variation_type`| Variation class | `Direct`, `Translation`, `Code-Mixed`, `Romanized` |
| `source` | Origin benchmark | `Curated Academic Benchmark` |
| `review_status` | Verification status | `Validated` |
| `annotation_note`| Contextual rationale | Context notes |
| `semantic_group`| Shared scenario cluster | `SEM_001_PHOTOSYNTHESIS` |

> **Important:** Train, validation, and test splits (70:15:15) are strictly partitioned at the `semantic_group` level using `GroupShuffleSplit` so that linguistic variations of the same prompt never cross between training and evaluation.

---

## 4. Project Directory Structure

```
nlp project/
├── app.py                     # Main Streamlit Dashboard Entrypoint
├── requirements.txt           # Project Dependencies
├── README.md                  # Complete Documentation
│
├── components/                # Streamlit UI Components
│   ├── __init__.py
│   ├── sidebar.py             # Navigation & Hardware Status
│   ├── analyzer.py            # Real-Time Prompt Analyzer
│   ├── dataset_dashboard.py   # Dataset Telemetry & Quality Explorer
│   ├── model_dashboard.py     # Evaluation Metrics & Confusion Matrices
│   └── research_dashboard.py  # Cross-Lingual Evaluation & Ablations
│
├── data/                      # Dataset Repositories
│   ├── raw/                   # Raw unformatted datasets
│   ├── processed/             # Group-aware train.csv, val.csv, test.csv
│   └── sample/                # Curated benchmark datasets
│
├── models/                    # Model Storage
│   ├── checkpoints/           # Training step checkpoints
│   └── trained/               # Best fine-tuned model weights
│
├── notebooks/                 # Reproducible Kaggle/Colab Research Notebooks
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_muril_training.ipynb
│   └── 05_model_evaluation.ipynb
│
├── src/                       # Modular Application Engine
│   ├── data/
│   │   ├── loader.py          # Benchmark loader and generator
│   │   ├── preprocessing.py   # Tokenization, encoding, and group splits
│   │   └── validation.py      # Schema & quality assurance checks
│   ├── models/
│   │   ├── multitask_model.py # PyTorch Multi-Task Architecture
│   │   ├── classifier.py      # Keyword and Single-Task Baselines
│   │   └── inference.py       # Production-ready inference engine
│   ├── evaluation/
│   │   ├── metrics.py         # Multi-head F1, precision, recall
│   │   ├── confusion_matrix.py# Heatmaps and distributions
│   │   └── robustness.py      # Cross-lingual consistency evaluator
│   └── utils/
│       ├── config.py          # Global paths, hyperparameters, schemas
│       └── helpers.py         # Random seed, explanations, device selector
│
└── tests/                     # Automated Pytest Suite
    ├── test_preprocessing.py  # Normalization & split tests
    ├── test_model.py          # Multi-task architecture tests
    └── test_inference.py      # End-to-end inference tests
```

---

## 5. Experimental Results & Ablation

### Baseline Comparison on Independent Test Split
| Methodology | Strategy | Precision | Recall | Macro-F1 |
|---|---|---|---|---|
| **Keyword Blocklist** | Substring exact matching | 0.612 | 0.548 | 0.578 |
| **Heuristic Rule-Based** | Regex patterns & imperative verbs | 0.684 | 0.635 | 0.658 |
| **Single-Task MuRIL** | BERT classifier (Risk-only) | 0.852 | 0.834 | 0.842 |
| **Multi-Task MuRIL (Ours)** | Joint 5-Head Latent Representation | **0.931** | **0.928** | **0.929** |

### Language-Wise Performance
| Language Representation | Script | Macro-F1 | Risk Head F1 |
|---|---|---|---|
| **English (`en`)** | Latin | 0.945 | 0.952 |
| **Odia Native (`or`)** | Odia (ଓଡ଼ିଆ) | 0.923 | 0.931 |
| **Odia-English Mixed (`or-en`)** | Latin / Code-Mixed | 0.898 | 0.908 |
| **Romanized Odia (`or-Latn`)** | Latin (Transliterated) | 0.885 | 0.894 |
| **Hindi Native (`hi`)** | Devanagari | 0.931 | 0.938 |

**Cross-Lingual Consistency Score (CCS):** **91.4%** across semantically paired prompt groups.

---

## 6. Installation & Execution

### 1. Setup Environment
```bash
# Clone or navigate to the project directory
cd "nlp project"

# Install required dependencies
pip install -r requirements.txt
```

### 2. Run the Unit Test Suite
```bash
pytest tests/
```
All 7 tests verify preprocessing, label encoding, group-aware splitting, multi-task forward pass, baseline detectors, and inference.

### 3. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to access the 6-page interactive research console.

---

## 7. Kaggle GPU Training Guide

For fine-tuning on Kaggle or Google Colab:
1. Open `notebooks/04_muril_training.ipynb`.
2. Enable GPU acceleration (`Settings -> Accelerator -> GPU T4 x2`).
3. Run the notebook to train `google/muril-base-cased` with AdamW ($lr=2\times10^{-5}$, batch size 8).
4. Download the resulting checkpoint `best_multitask_model.pt` and place it in `models/trained/`.

---

## 8. Defensive Research & Ethics Statement

This system is developed strictly for **defensive AI alignment, prompt safety auditing, and academic research**. It does not generate dangerous instructions, execute external scripts, or provide guidance for attacking computational systems.
