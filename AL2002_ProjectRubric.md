**AL2002 Artificial Intelligence BS (CS) Spring 2026**

Lab Project

# Intelligent Reading Comprehension and Quiz Generation System using Machine Learning 

**Project Evaluation Rubric Sheet**

Evaluation Date: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\
Evaluation Instructor: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

  -------------------------------- ------------------------------------------------------------------ --------------------------------
  Reg. No. #:                      Student Name:                                                      Section:
  \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_   \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_   \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

  Reg. No. #:                      Student Name:                                                      Section:
  \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_   \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_   \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
  -------------------------------- ------------------------------------------------------------------ --------------------------------

+------------------+-----+-----------------------+----------+----------+
| **Component**    | **M | **Key Evaluation      | *        | **Marks  |
|                  | ark | Criteria**            | *Self-Ev | A        |
|                  | s** |                       | aluation | warded** |
|                  |     |                       | Marks**  |          |
+==================+=====+=======================+==========+==========+
| **1. Data        |     |                       |          |          |
| Analysis &       |     |                       |          |          |
| Preprocessing    |     |                       |          |          |
| \[10 Marks\]**   |     |                       |          |          |
+------------------+-----+-----------------------+----------+----------+
| Exploratory Data | 3   | Data overview, handle |          |          |
| Analysis         |     | missing value         |          |          |
|                  |     | analysis, statistical |          |          |
|                  |     | analysis, outliers    |          |          |
|                  |     | detection             |          |          |
+------------------+-----+-----------------------+----------+----------+
| Visualizations   | 3   | Data distribution     |          |          |
|                  |     | analysis, correlation |          |          |
|                  |     | analysis, feature     |          |          |
|                  |     | relationship          |          |          |
+------------------+-----+-----------------------+----------+----------+
| Preprocessing    | 4   | Lowercasing,          |          |          |
| Pipeline         |     | punctuation removal;  |          |          |
|                  |     | Encoding categorical  |          |          |
|                  |     | data: One-Hot         |          |          |
|                  |     | Encoding features     |          |          |
|                  |     | saved or other        |          |          |
|                  |     | technique, Feature    |          |          |
|                  |     | scaling, Data         |          |          |
|                  |     | Cleaning, Feature     |          |          |
|                  |     | selection, Feature    |          |          |
|                  |     | Engineering, Handling |          |          |
|                  |     | Imbalace data (If     |          |          |
|                  |     | present), Data        |          |          |
|                  |     | Transormation,        |          |          |
|                  |     | Train-Test splits     |          |          |
+------------------+-----+-----------------------+----------+----------+
| **2. Model A --- |     |                       |          |          |
| Traditional ML   |     |                       |          |          |
| \[20 Marks\]**   |     |                       |          |          |
+------------------+-----+-----------------------+----------+----------+
| ≥ 2 Classifiers  | 5   | Implementation of at  |          |          |
| Implemented      |     | least two traditional |          |          |
|                  |     | machine learning      |          |          |
|                  |     | classifiers such as   |          |          |
|                  |     | Logistic Regression,  |          |          |
|                  |     | SVM, Naive Bayes,     |          |          |
|                  |     | Random Forest, or     |          |          |
|                  |     | XGBoost               |          |          |
+------------------+-----+-----------------------+----------+----------+
| Feature          | 5   | Appropriate feature   |          |          |
| Engineering      |     | engineering           |          |          |
|                  |     | techniques applied,   |          |          |
|                  |     | such as One-Hot       |          |          |
|                  |     | Encoding, TF-IDF, Bag |          |          |
|                  |     | of Words, cosine      |          |          |
|                  |     | similarity, or other  |          |          |
|                  |     | relevant techniques   |          |          |
|                  |     | depending on the      |          |          |
|                  |     | dataset               |          |          |
+------------------+-----+-----------------------+----------+----------+
| Question         | 5   | Meaningful question   |          |          |
| Generation &     |     | generation, correct   |          |          |
| Answer           |     | answer                |          |          |
| Verification     |     | pre                   |          |          |
|                  |     | diction/verification. |          |          |
|                  |     | Given (article,       |          |          |
|                  |     | question, option)     |          |          |
|                  |     | triple, model         |          |          |
|                  |     | predicts whether      |          |          |
|                  |     | selected option is    |          |          |
|                  |     | correct.              |          |          |
+------------------+-----+-----------------------+----------+----------+
| Metric           | 5   | Performance metrics   |          |          |
| Comparison Table |     | reported and compared |          |          |
|                  |     | across all models on  |          |          |
|                  |     | validation/test set   |          |          |
|                  |     | using appropriate     |          |          |
|                  |     | evaluation metrics    |          |          |
|                  |     | such as Accuracy,     |          |          |
|                  |     | Precision, Recall,    |          |          |
|                  |     | F1-Score, Confusion   |          |          |
|                  |     | Matrix, or            |          |          |
|                  |     | BLEU/ROUGE/METEOR for |          |          |
|                  |     | NLP generation tasks  |          |          |
+------------------+-----+-----------------------+----------+----------+
| **3. Model A --- |     |                       |          |          |
| Unsupervised /   |     |                       |          |          |
| Semi-Supervised  |     |                       |          |          |
| \[10 Marks\]**   |     |                       |          |          |
+------------------+-----+-----------------------+----------+----------+
| Approach         | 5   | Implementation and    |          |          |
| Implemented      |     | evaluation of at      |          |          |
|                  |     | least one             |          |          |
|                  |     | unsupervised or       |          |          |
|                  |     | semi-supervised       |          |          |
|                  |     | approach such as      |          |          |
|                  |     | K-Means, Label        |          |          |
|                  |     | Propagation, or GMM;  |          |          |
|                  |     | comparison against    |          |          |
|                  |     | supervised models     |          |          |
+------------------+-----+-----------------------+----------+----------+
| Evaluation       | 5   | Clustering purity,    |          |          |
| Metrics          |     | silhouette score, or  |          |          |
|                  |     | semi-supervised F1    |          |          |
|                  |     | reported              |          |          |
+------------------+-----+-----------------------+----------+----------+
| **4. Model A --- |     |                       |          |          |
| Ensemble \[05    |     |                       |          |          |
| Marks\]**        |     |                       |          |          |
+------------------+-----+-----------------------+----------+----------+
| Ensemble         | 3   | Soft voting (average  |          |          |
| Strategy         |     | probability outputs   |          |          |
|                  |     | from SVM + LR + NB),  |          |          |
|                  |     | Hard Voting (majority |          |          |
|                  |     | vote across ≥ 3       |          |          |
|                  |     | classifiers), or      |          |          |
|                  |     | Stacking              |          |          |
|                  |     | (meta-classifier      |          |          |
|                  |     | trained on base-model |          |          |
|                  |     | outputs). Code is     |          |          |
|                  |     | clean and strategy is |          |          |
|                  |     | clearly named.        |          |          |
+------------------+-----+-----------------------+----------+----------+
| Improvement      | 2   | Ensemble outperforms  |          |          |
| Demonstrated     |     | (or matches with      |          |          |
|                  |     | justification)        |          |          |
|                  |     | individual models     |          |          |
+------------------+-----+-----------------------+----------+----------+
| **5. Model B --- |     |                       |          |          |
| Distractor       |     |                       |          |          |
| Generation \[20  |     |                       |          |          |
| Marks\]**        |     |                       |          |          |
+------------------+-----+-----------------------+----------+----------+
| Candidate        | 4   | Phrases extracted     |          |          |
| Extraction       |     | from passage via      |          |          |
| Pipeline         |     | string matching or    |          |          |
|                  |     | frequency-based       |          |          |
|                  |     | selection             |          |          |
+------------------+-----+-----------------------+----------+----------+
| Feature          | 4   | For each candidate:   |          |          |
| Engineering for  |     | (a) One-Hot Encoding  |          |          |
| Ranking          |     | cosine similarity to  |          |          |
|                  |     | correct answer        |          |          |
|                  |     | computed, (b)         |          |          |
|                  |     | character-level match |          |          |
|                  |     | score computed, (c)   |          |          |
|                  |     | passage frequency     |          |          |
|                  |     | recorded. TF-IDF      |          |          |
|                  |     | cosine similarity     |          |          |
|                  |     | used optionally and   |          |          |
|                  |     | noted.                |          |          |
+------------------+-----+-----------------------+----------+----------+
| ML Ranker for    | 5   | Logistic Regression   |          |          |
| Distractors      |     | or Random Forest      |          |          |
|                  |     | trained to score each |          |          |
|                  |     | candidate; top-3      |          |          |
|                  |     | non-answer candidates |          |          |
|                  |     | selected as           |          |          |
|                  |     | distractors;          |          |          |
|                  |     | diversity penalty     |          |          |
|                  |     | applied so            |          |          |
|                  |     | distractors are not   |          |          |
|                  |     | trivially similar;    |          |          |
|                  |     | model persisted via   |          |          |
|                  |     | joblib.               |          |          |
+------------------+-----+-----------------------+----------+----------+
| Plausibility &   | 4   | Each generated quiz   |          |          |
| Diversity        |     | presents three        |          |          |
|                  |     | distractors per       |          |          |
|                  |     | question; diverse,    |          |          |
|                  |     | grammatically         |          |          |
|                  |     | consistent, factually |          |          |
|                  |     | wrong.                |          |          |
|                  |     |                       |          |          |
|                  |     | Example (a) appear    |          |          |
|                  |     | plausible to an       |          |          |
|                  |     | uninformed reader,    |          |          |
|                  |     | (b) are definitively  |          |          |
|                  |     | wrong w.r.t. the      |          |          |
|                  |     | passage, (c) are      |          |          |
|                  |     | lexically diverse,    |          |          |
|                  |     | and (d) share the     |          |          |
|                  |     | same syntactic form   |          |          |
|                  |     | as the correct        |          |          |
|                  |     | answer.               |          |          |
+------------------+-----+-----------------------+----------+----------+
| Evaluation       | 3   | BLEU, ROUGE, METEOR   |          |          |
| (BL              |     | scores reported for   |          |          |
| EU/ROUGE/METEOR) |     | distractor generation |          |          |
|                  |     | quality               |          |          |
+------------------+-----+-----------------------+----------+----------+
| **6. Model B --- |     |                       |          |          |
| Hint Generation  |     |                       |          |          |
| \[10 Marks\]**   |     |                       |          |          |
+------------------+-----+-----------------------+----------+----------+
| Extractive Hint  | 5   | Each sentence in the  |          |          |
| Scorer           |     | passage scored by     |          |          |
|                  |     | relevance to the      |          |          |
|                  |     | question using; (a)   |          |          |
|                  |     | cosine similarity of  |          |          |
|                  |     | One-Hot Encoded /     |          |          |
|                  |     | sentence-embedding    |          |          |
|                  |     | representations, OR   |          |          |
|                  |     | (b) Logistic          |          |          |
|                  |     | Regression trained on |          |          |
|                  |     | keyword overlap,      |          |          |
|                  |     | sentence position,    |          |          |
|                  |     | and sentence-length   |          |          |
|                  |     | features. Top-K       |          |          |
|                  |     | sentences surfaced as |          |          |
|                  |     | ranked hints.         |          |          |
+------------------+-----+-----------------------+----------+----------+
| Graduated Hints  | 5   | Hint 1 = general,     |          |          |
| (3 Levels)       |     | Hint 2 = specific,    |          |          |
|                  |     | Hint 3 =              |          |          |
|                  |     | near-explicit         |          |          |
|                  |     | (sentence containing  |          |          |
|                  |     | or closely            |          |          |
|                  |     | paraphrasing the      |          |          |
|                  |     | answer). All three    |          |          |
|                  |     | levels present and    |          |          |
|                  |     | correctly ordered in  |          |          |
|                  |     | the UI.               |          |          |
+------------------+-----+-----------------------+----------+----------+
| **7. User        |     |                       |          |          |
| Interface \[15   |     |                       |          |          |
| Marks\]**        |     |                       |          |          |
+------------------+-----+-----------------------+----------+----------+
| Screen 1 ---     | 3   | Text area for pasting |          |          |
| Article Input    |     | or uploading a        |          |          |
|                  |     | reading passage;      |          |          |
|                  |     | option to load a      |          |          |
|                  |     | random RACE dataset   |          |          |
|                  |     | sample for quick      |          |          |
|                  |     | testing; \'Submit\'   |          |          |
|                  |     | button triggers both  |          |          |
|                  |     | Model A and Model B   |          |          |
|                  |     | inference             |          |          |
|                  |     | simultaneously;       |          |          |
|                  |     | loading indicator     |          |          |
|                  |     | shown during          |          |          |
|                  |     | inference.            |          |          |
+------------------+-----+-----------------------+----------+----------+
| Screen 2 ---     | 4   | Generated question    |          |          |
| Quiz View        |     | displayed on screen;  |          |          |
|                  |     | 4 options (A--D);     |          |          |
|                  |     | Check button;         |          |          |
|                  |     | colour-coded          |          |          |
|                  |     | correct/incorrect     |          |          |
|                  |     | (proper and           |          |          |
|                  |     | functional).          |          |          |
+------------------+-----+-----------------------+----------+----------+
| Screen 3 ---     | 4   | Collapsible or tabbed |          |          |
| Hint Panel       |     | panel with three      |          |          |
|                  |     | graduated hints from  |          |          |
|                  |     | Model B; hints        |          |          |
|                  |     | revealed              |          |          |
|                  |     | progressively;        |          |          |
|                  |     | \'Reveal Answer\'     |          |          |
|                  |     | button appears only   |          |          |
|                  |     | after all hints have  |          |          |
|                  |     | been viewed; UI       |          |          |
|                  |     | prevents skipping     |          |          |
|                  |     | hints.                |          |          |
+------------------+-----+-----------------------+----------+----------+
| Screen 4 ---     | 2   | Model metrics         |          |          |
| Analytics        |     | displayed; inference  |          |          |
| Dashboard        |     | latency shown; CSV    |          |          |
|                  |     | export available      |          |          |
+------------------+-----+-----------------------+----------+----------+
| UX & Error       | 2   | All four screens      |          |          |
| Handling         |     | usable without        |          |          |
|                  |     | reading a manual;     |          |          |
|                  |     | friendly error        |          |          |
|                  |     | messages for empty    |          |          |
|                  |     | input and model       |          |          |
|                  |     | failure; loading      |          |          |
|                  |     | indicators during     |          |          |
|                  |     | inference; sufficient |          |          |
|                  |     | colour contrast (WCAG |          |          |
|                  |     | AA) and readable font |          |          |
|                  |     | sizes; keyboard       |          |          |
|                  |     | navigation possible.  |          |          |
+------------------+-----+-----------------------+----------+----------+
| **8. Final       |     |                       |          |          |
| Report \[05      |     |                       |          |          |
| Marks\]**        |     |                       |          |          |
+------------------+-----+-----------------------+----------+----------+
| Structure &      | 3   | All 11 required       |          |          |
| Completeness     |     | sections present:     |          |          |
|                  |     | Abstract (≤200        |          |          |
|                  |     | words), Introduction  |          |          |
|                  |     | & Motivation, Related |          |          |
|                  |     | Work (≥5 cited papers |          |          |
|                  |     | including RACE, BERT, |          |          |
|                  |     | BLEU/ROUGE            |          |          |
|                  |     | originals), Dataset   |          |          |
|                  |     | Analysis, Model A     |          |          |
|                  |     | Des                   |          |          |
|                  |     | ign/Training/Results, |          |          |
|                  |     | Model B               |          |          |
|                  |     | Des                   |          |          |
|                  |     | ign/Training/Results, |          |          |
|                  |     | UI Description,       |          |          |
|                  |     | Evaluation &          |          |          |
|                  |     | Discussion,           |          |          |
|                  |     | Limitations & Future  |          |          |
|                  |     | Work, Conclusion,     |          |          |
|                  |     | References.           |          |          |
|                  |     | Paper-style           |          |          |
|                  |     | formatting.           |          |          |
+------------------+-----+-----------------------+----------+----------+
| Clarity &        | 2   | Methodology clear;    |          |          |
| Discussion       |     | results discussed;    |          |          |
|                  |     | limitations and work  |          |          |
|                  |     | addressed             |          |          |
+------------------+-----+-----------------------+----------+----------+
| **9. Code        |     |                       |          |          |
| Quality \[05     |     |                       |          |          |
| Marks\]**        |     |                       |          |          |
+------------------+-----+-----------------------+----------+----------+
| Project          | 5   | Correct Submission of |          |          |
| Submission       |     | project folder.       |          |          |
+------------------+-----+-----------------------+----------+----------+
| **TOTAL**        |     |                       | **100**  | **/      |
|                  |     |                       |          | 100**    |
+------------------+-----+-----------------------+----------+----------+

**Key Suggestions (Applied to This Evaluation)**

-   **Checkpoints Required:** Groups using Google Colab or other
    platforms implement model checkpoints.

-   **Evaluation Metrics Update:** Not necessary to use Accuracy or
    Precision as primary metrics for generation tasks. Use **BLEU Score,
    ROUGE Score, and METEOR Score** for Model A and Model B evaluation.

-   Before coming for the demo, kindly fill out this sheet. Also, **make
    sure to arrive at the demo venue 5 minutes before your scheduled
    time**. After arriving, directly download your submission from GCR
    and keep it ready in a working state for the demo.

  ---------------------------------------------------------------- -----------------------------------
  Evaluator Signature:                                             **Final Marks Awarded:
  \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_   \_\_\_\_\_\_\_ / 100**

  Remarks:                                                         
  ---------------------------------------------------------------- -----------------------------------
