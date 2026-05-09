# AI Lab Project — Agent Context Document

## Project Title
Intelligent Reading Comprehension and Quiz Generation System using Classical Machine Learning

---

# 1. Project Overview

This project is an AI-powered Reading Comprehension (RC) system built on the RACE dataset.

The system must:

1. Accept a reading passage/article.
2. Generate or retrieve comprehension questions.
3. Generate answer options.
4. Predict/verify the correct answer.
5. Generate distractor options.
6. Generate hints.
7. Expose all functionality through a UI.

The project emphasizes:

- Classical Machine Learning
- TF-IDF
- Cosine Similarity
- Feature Engineering
- scikit-learn pipelines
- Modular software architecture

Large transformer models are NOT necessary.

---

# 2. Dataset Information

## Dataset Name
RACE (ReAding Comprehension from Examinations)

## Dataset Structure
Each row contains:

| Column | Description |
|---|---|
| id | Unique sample ID |
| article | Reading passage |
| question | Multiple-choice question |
| A | Option A |
| B | Option B |
| C | Option C |
| D | Option D |
| answer | Correct answer label |

## Approximate Dataset Size

- 28,000+ passages
- ~100,000 questions

## Expected CSV Files

- train.csv
- val.csv
- test.csv

---

# 3. System Architecture

The project is divided into 3 layers:

## Layer 1 — Data Layer
Responsible for:

- Dataset loading
- Preprocessing
- Feature engineering
- Train/validation/test split handling
- TF-IDF vectorization
- Similarity computation

## Layer 2 — Model Layer
Contains:

### Model A
Question/Answer verification and question generation.

### Model B
Distractor generation and hint generation.

## Layer 3 — UI Layer
Implemented separately.

The ML/backend layer should expose clean inference functions callable by the UI.

---

# 4. Recommended Project Structure

```text
race_rc_project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── model_a/
│   └── model_b/
│
├── src/
│   ├── preprocessing.py
│   ├── model_a_train.py
│   ├── model_b_train.py
│   ├── inference.py
│   ├── evaluate.py
│   └── utils.py
│
├── notebooks/
│   ├── EDA.ipynb
│   └── experiments.ipynb
│
├── tests/
│   └── test_inference.py
│
├── ui/
│   └── app.py
│
├── requirements.txt
├── README.md
└── report/
```

---

# 5. Core ML Concepts

## 5.1 TF-IDF

TF-IDF measures word importance.

### Formula

TF-IDF(t,d,D) = TF(t,d) × IDF(t,D)

Where:

- TF = term frequency in current document
- IDF = inverse document frequency across corpus

### Recommended scikit-learn Parameters

```python
TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    sublinear_tf=True,
    ngram_range=(1,2),
    min_df=2,
    max_df=0.95
)
```

## 5.2 Cosine Similarity

Used for:

- sentence ranking
- answer verification
- distractor selection
- hint generation

### Formula

cos(theta) = (A · B) / (||A|| × ||B||)

### scikit-learn Usage

```python
from sklearn.metrics.pairwise import cosine_similarity
```

---

# 6. Model A — Question & Answer Verification

## 6.1 Main Objective

Given:

- article
- question
- answer option

Predict:

- whether the option is correct or incorrect.

This is treated as a binary classification problem.

---

# 6.2 Training Data Construction

For each question, generate 4 samples.

Example:

| Option | Label |
|---|---|
| A | 0 |
| B | 1 |
| C | 0 |
| D | 0 |

Where:

- 1 = correct option
- 0 = incorrect option

---

# 6.3 Recommended Input Representation

The recommended text combination is:

```python
combined = article + article + question + option
```

Repeating the article gives more weight to passage information.

---

# 6.4 Recommended Models

At least TWO classical ML models should be implemented.

## Strongly Recommended

### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
```

### Support Vector Machine (SVM)

```python
from sklearn.svm import LinearSVC
```

## Optional

- Naive Bayes
- Random Forest
- XGBoost

---

# 6.5 Suggested Training Pipeline

## Step 1 — Build combined text

```python
combined_text = f"{article} {article} {question} {option}"
```

## Step 2 — Vectorization

```python
X_train = vectorizer.fit_transform(train_texts)
```

## Step 3 — Train model

```python
model.fit(X_train, y_train)
```

## Step 4 — Evaluate

```python
preds = model.predict(X_val)
```

---

# 6.6 Evaluation Metrics

Required metrics:

- Accuracy
- Macro F1
- Exact Match
- Confusion Matrix

Useful imports:

```python
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)
```

---

# 7. Model A — Question Generation

This project does NOT require neural text generation.

A template-based approach is acceptable.

---

# 7.1 Recommended Workflow

## Step 1 — Sentence Extraction

Select the most relevant sentence from the article.

Possible methods:

- cosine similarity
- keyword overlap
- TF-IDF relevance

## Step 2 — Template Conversion

Convert selected sentence into a question.

Example:

Sentence:

```text
Ali went to Lahore in 2020.
```

Question:

```text
Where did Ali go in 2020?
```

---

# 7.2 Simplified Implementation Strategy

A practical solution:

1. Split article into sentences.
2. Compute similarity between sentence and answer/question.
3. Select top sentence.
4. Apply simple Wh-word templates.

This is sufficient for the project scope.

---

# 8. Unsupervised / Semi-Supervised Component

At least ONE unsupervised or semi-supervised approach is mandatory.

Recommended easiest option:

## K-Means Clustering

```python
from sklearn.cluster import KMeans
```

Apply K-Means on TF-IDF vectors.

### Suggested Evaluation

- silhouette score
- cluster purity

Alternative options:

- Label Propagation
- Gaussian Mixture Models

---

# 9. Model A Ensemble Strategy

Optional but recommended.

Combine predictions from:

- Logistic Regression
- SVM

Strategies:

- hard voting
- soft voting
- stacking

---

# 10. Model B — Distractor Generation

## Objective

Given:

- article
- question
- correct answer

Generate:

- 3 plausible incorrect options.

---

# 10.1 Recommended Distractor Pipeline

## Step 1 — Candidate Extraction

Extract candidate phrases from article.

Possible methods:

- high-frequency words
- noun phrases
- named entities
- frequent content words

Even simple frequency-based extraction is acceptable.

---

## Step 2 — Similarity Computation

Compute:

- TF-IDF cosine similarity
- lexical overlap
- frequency score

---

## Step 3 — Candidate Selection

Select candidates that are:

- semantically related
- not identical to answer
- grammatically plausible

A useful heuristic:

- choose medium-similarity candidates.

---

# 10.2 Simplified Distractor Strategy

```python
1. Extract candidate phrases.
2. Vectorize candidates + answer.
3. Compute cosine similarity.
4. Rank by similarity.
5. Remove duplicates.
6. Select top distractors.
```

---

# 11. Model B — Hint Generation

## Objective

Generate graduated hints that guide users toward the answer.

---

# 11.1 Recommended Hint Strategy

## Step 1

Split article into sentences.

## Step 2

Compute similarity between:

- question
- each sentence

## Step 3

Rank sentences.

## Step 4

Use ranked sentences as hints.

Example:

| Hint | Description |
|---|---|
| Hint 1 | low-detail clue |
| Hint 2 | medium-detail clue |
| Hint 3 | near-explicit clue |

---

# 12. Recommended Backend API

The backend should expose clean functions callable by the UI.

## Suggested Functions

```python
def predict_answer(article, question, options):
    pass


def generate_question(article):
    pass


def generate_distractors(article, question, answer):
    pass


def generate_hints(article, question):
    pass
```

---

# 13. Model Persistence

Save BOTH:

- trained models
- fitted vectorizers

Recommended library:

```python
import joblib
```

## Save

```python
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
```

## Load

```python
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
```

---

# 14. Critical Warnings

## 14.1 Data Leakage

ONLY use:

```python
fit_transform()
```

on training data.

Use:

```python
transform()
```

for validation and test data.

Incorrect usage causes artificially inflated metrics.

---

# 14.2 Sparse Matrices

DO NOT convert large TF-IDF matrices to dense arrays.

Avoid:

```python
X.toarray()
```

Use sparse matrices directly.

---

# 14.3 Save Vectorizer

A trained model without its fitted vectorizer is unusable.

Always save both.

---

# 15. Recommended Libraries

## Core ML

```python
scikit-learn
```

## Data Handling

```python
pandas
numpy
```

## Model Persistence

```python
joblib
```

## NLP Utilities

```python
nltk
```

## Optional

```python
gensim
xgboost
```

---

# 16. Recommended Development Plan

## Phase 1 — EDA & Preprocessing

Tasks:

- inspect dataset
- clean text
- tokenize
- build training samples

Deliverables:

- EDA notebook
- preprocessing.py

---

## Phase 2 — Verification Models

Tasks:

- TF-IDF vectorization
- Logistic Regression
- SVM
- evaluation metrics

Deliverables:

- trained models
- evaluation tables

---

## Phase 3 — Unsupervised Component

Tasks:

- K-Means clustering
- silhouette score

---

## Phase 4 — Distractor Generation

Tasks:

- candidate extraction
- similarity ranking
- distractor selection

---

## Phase 5 — Hint Generation

Tasks:

- sentence ranking
- graduated hints

---

## Phase 6 — Unified Inference Layer

Tasks:

- inference.py
- reusable prediction functions

---

## Phase 7 — Integration

Tasks:

- connect backend to UI
- error handling
- save/load models

---

# 17. Minimum Viable Solution

A strong project submission can be achieved with:

- TF-IDF vectorization
- Logistic Regression
- SVM
- K-Means clustering
- cosine similarity sentence ranking
- heuristic distractor generation
- extractive hint generation
- modular backend API

No transformer models are required.

---

# 18. Expected Deliverables

## Required Files

- README.md
- requirements.txt
- EDA notebook
- trained models
- source code
- final report

## Required Demonstrations

- working inference pipeline
- UI integration
- evaluation metrics
- reproducible training

---

# 19. Recommended requirements.txt

```text
pandas
numpy
scikit-learn
nltk
joblib
matplotlib
seaborn
plotly
streamlit
```

---

# 20. Final Practical Guidance

This project is primarily a:

- TF-IDF
- cosine similarity
- classical ML
- ranking system

A clean, modular, reliable implementation is significantly better than an overcomplicated neural approach.

Priority should be:

1. Correct preprocessing
2. Proper TF-IDF usage
3. Good evaluation
4. Reliable inference pipeline
5. Clean modular architecture
6. Easy UI integration

