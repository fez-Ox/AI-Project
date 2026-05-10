# AI Reading Comprehension & Quiz Generation System

An AI-powered Reading Comprehension and Quiz Generation System built on the **RACE Dataset**. The system generates comprehension questions, predicts correct answers, creates distractor options, and provides graduated hints — all through a Streamlit web interface.

## Features

- **Model A** — Question & Answer Generator / Verifier (LR, SVM, Naive Bayes, Ensemble)
- **Model B** — Distractor & Hint Generator (Word2Vec, ML-ranked distractors, ML-scored hints)
- **UI** — 4-screen Streamlit app (Article Input, Quiz, Hints, Developer Dashboard)

## Setup

```bash
# Clone the repository
git clone https://github.com/fez-Ox/AI-Project.git
cd AI-Project

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Dataset

Download the RACE dataset from [Kaggle](https://www.kaggle.com/datasets/ankitdhiman7/race-dataset) and place the CSV files in `data/raw/`:

```
data/raw/
├── train.csv
├── dev.csv
└── test.csv
```

## Training

Run the full pipeline in order:

```bash
# Step 1: Preprocess raw data
python src/preprocessing.py

# Step 2: Train Model A (LR, SVM, NB, Ensemble)
python src/model_a_train.py

# Step 3: Train Model B (Word2Vec, Distractor Ranker, Hint Scorer)
python src/model_b_train.py

# Step 4: Evaluate models
python src/evaluate.py
```

Or run everything from the provided Colab notebook: `AI_Project_Training.ipynb`

## Running the UI

```bash
streamlit run ui/app.py
```

## Running Tests

```bash
python -m pytest tests/
```

## Project Structure

```
AI-Project/
├── data/
│   ├── raw/                    # Original RACE CSV files
│   └── processed/              # Preprocessed binary classification data
├── models/
│   ├── model_a/
│   │   ├── neural/             # (Reserved for neural model checkpoints)
│   │   └── traditional/        # Pickled sklearn models
│   └── model_b/
│       ├── neural/             # (Reserved for neural model checkpoints)
│       └── traditional/        # Word2Vec, distractor ranker, hint scorer
├── src/
│   ├── preprocessing.py        # Dataset loading & feature engineering
│   ├── model_a_train.py        # Training script for Model A
│   ├── model_b_train.py        # Training script for Model B
│   ├── inference.py            # Unified inference API
│   ├── evaluate.py             # Metric computation
│   └── utils.py                # Shared utility functions
├── ui/
│   ├── app.py                  # Streamlit entry point
│   └── components/             # Reusable UI helpers
├── notebooks/
│   ├── EDA.ipynb               # Exploratory Data Analysis
│   └── experiments.ipynb       # Experiment tracking
├── tests/
│   ├── test_inference.py       # Unit tests
│   └── run_examples.py         # Example inference runner
├── report/
│   └── final_report.md         # Final report (Markdown)
├── requirements.txt
└── README.md
```

## Tech Stack

| Category | Library |
|----------|---------|
| Classical ML | scikit-learn |
| Text Features | CountVectorizer (One-Hot), TfidfVectorizer |
| Word Embeddings | Gensim (Word2Vec) |
| Data | pandas, numpy |
| Evaluation | scikit-learn metrics |
| UI | Streamlit |
| Visualization | matplotlib, seaborn, plotly |

## Team

BS(CS) Spring 2026 — FAST NUCES Islamabad

## License

For academic use only.
