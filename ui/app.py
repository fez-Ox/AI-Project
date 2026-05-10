import os
import sys
import time

import pandas as pd
import streamlit as st

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference import (
    generate_distractors,
    generate_hints,
    generate_question,
    predict_answer,
)
from ui.components import load_random_sample, measure_latency, session_log_to_csv

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(page_title="AI Reading Comprehension", layout="wide")
st.title("📚 Intelligent Reading Comprehension System")
st.caption("⚠️ All questions and answers are AI-generated. Errors are possible.")

# ── Session State Defaults ───────────────────────────────────
for key, default in {
    "article": "",
    "question": "",
    "options": {},
    "correct_answer": "",
    "distractors": [],
    "hints_used": 0,
    "quiz_submitted": False,
    "selected_option": None,
    "session_log": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
use_ensemble = st.sidebar.checkbox("Use Ensemble Model", value=True)

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📝 Article Input",
    "❓ Quiz",
    "💡 Hints",
    "📊 Dashboard",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — Article Input
# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter or Load a Reading Passage")

    # Option to load a random RACE sample
    col_load, col_clear = st.columns(2)
    with col_load:
        if st.button("🎲 Load Random RACE Sample"):
            sample = load_random_sample()
            if sample:
                st.session_state["article"] = sample["article"]
                st.session_state["question"] = sample["question"]
                st.session_state["options"] = sample["options"]
                st.session_state["correct_answer"] = sample["answer"]
                st.rerun()
            else:
                st.error("Raw dataset not found. Place CSV files in data/raw/.")

    with col_clear:
        if st.button("🗑️ Clear All"):
            for key in ["article", "question", "options", "correct_answer",
                        "distractors", "hints_used", "quiz_submitted", "selected_option"]:
                st.session_state[key] = "" if isinstance(st.session_state[key], str) else type(st.session_state.get(key, None))()
            st.session_state["hints_used"] = 0
            st.session_state["quiz_submitted"] = False
            st.rerun()

    # Text area for article input
    article_text = st.text_area(
        "Reading Passage:",
        value=st.session_state["article"],
        height=200,
        placeholder="Paste a reading passage here, or click 'Load Random RACE Sample'...",
    )

    # File upload
    uploaded_file = st.file_uploader("Or upload a text file:", type=["txt"])
    if uploaded_file:
        article_text = uploaded_file.read().decode("utf-8")

    # Submit button
    if st.button("🚀 Submit & Generate Quiz", type="primary"):
        if not article_text.strip():
            st.error("Please enter a reading passage first.")
        else:
            st.session_state["article"] = article_text
            st.session_state["quiz_submitted"] = False
            st.session_state["hints_used"] = 0

            with st.spinner("Generating question..."):
                question, q_time = measure_latency(generate_question, article_text)
                st.session_state["question"] = question

            with st.spinner("Generating answer and distractors..."):
                # Use the generated question to predict an answer from the passage
                # Then generate distractors for it
                try:
                    # Generate a reasonable answer (first key sentence)
                    import nltk
                    sents = nltk.sent_tokenize(article_text)
                    answer_text = sents[0] if sents else article_text[:50]

                    distractors, d_time = measure_latency(
                        generate_distractors, article_text, question, answer_text
                    )

                    # Build the MCQ options: correct answer + 3 distractors
                    import random
                    all_opts = [answer_text] + distractors
                    random.shuffle(all_opts)
                    correct_idx = all_opts.index(answer_text)
                    keys = ["A", "B", "C", "D"]

                    st.session_state["options"] = {
                        keys[i]: all_opts[i] for i in range(min(4, len(all_opts)))
                    }
                    st.session_state["correct_answer"] = keys[correct_idx]
                    st.session_state["distractors"] = distractors

                    # Log latency
                    st.session_state["session_log"].append({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "action": "generate_quiz",
                        "question_latency_s": round(q_time, 3),
                        "distractor_latency_s": round(d_time, 3),
                    })

                except FileNotFoundError as e:
                    st.error(f"Model not found: {e}. Did you run training?")
                except Exception as e:
                    st.error(f"Error: {e}")

            st.success("Quiz generated! Go to the **Quiz** tab.")

    # Show current article preview
    if st.session_state["article"]:
        st.markdown("---")
        st.markdown("**Current Passage Preview:**")
        st.info(st.session_state["article"][:500] + ("..." if len(st.session_state["article"]) > 500 else ""))


# ══════════════════════════════════════════════════════════════
# TAB 2 — Quiz View
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Take the Quiz")

    if not st.session_state["question"] or not st.session_state["options"]:
        st.warning("No quiz loaded. Go to the **Article Input** tab first.")
    else:
        st.markdown(f"**Question:** {st.session_state['question']}")
        st.markdown("---")

        options = st.session_state["options"]
        option_labels = [f"{k}. {v}" for k, v in options.items()]

        selected = st.radio("Select your answer:", option_labels, index=None)

        col_check, col_predict = st.columns(2)

        with col_check:
            if st.button("✅ Check My Answer"):
                if selected is None:
                    st.error("Please select an option first.")
                else:
                    chosen_key = selected.split(".")[0].strip()
                    correct_key = st.session_state["correct_answer"]

                    if chosen_key == correct_key:
                        st.success(f"🎉 Correct! The answer is **{correct_key}. {options[correct_key]}**")
                    else:
                        st.error(
                            f"❌ Incorrect. You chose **{chosen_key}**, "
                            f"but the correct answer is **{correct_key}. {options[correct_key]}**"
                        )

                    st.session_state["quiz_submitted"] = True

                    st.session_state["session_log"].append({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "action": "check_answer",
                        "chosen": chosen_key,
                        "correct": correct_key,
                        "is_correct": chosen_key == correct_key,
                    })

        with col_predict:
            if st.button("🤖 Let AI Predict"):
                with st.spinner("AI is analyzing..."):
                    try:
                        prediction, latency = measure_latency(
                            predict_answer,
                            st.session_state["article"],
                            st.session_state["question"],
                            options,
                            use_ensemble,
                        )

                        pred_key = prediction["predicted_option"]
                        correct_key = st.session_state["correct_answer"]

                        if pred_key == correct_key:
                            st.success(
                                f"🤖 AI predicts **{pred_key}. {prediction['predicted_text']}** — Correct! "
                                f"(took {latency:.2f}s)"
                            )
                        else:
                            st.warning(
                                f"🤖 AI predicts **{pred_key}. {prediction['predicted_text']}** — "
                                f"Correct was **{correct_key}** (took {latency:.2f}s)"
                            )

                        st.markdown("**Confidence Scores:**")
                        scores_df = pd.DataFrame({
                            "Option": list(prediction["confidence_scores"].keys()),
                            "Confidence": list(prediction["confidence_scores"].values()),
                        })
                        st.bar_chart(scores_df.set_index("Option"))

                        st.session_state["session_log"].append({
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "action": "ai_predict",
                            "predicted": pred_key,
                            "correct": correct_key,
                            "latency_s": round(latency, 3),
                        })

                    except FileNotFoundError as e:
                        st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════
# TAB 3 — Hint Panel
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Graduated Hints")

    if not st.session_state["question"]:
        st.warning("No quiz loaded. Go to the **Article Input** tab first.")
    else:
        st.markdown(f"**Question:** {st.session_state['question']}")
        st.markdown("---")

        # Generate hints once and cache
        if "hints" not in st.session_state or not st.session_state.get("hints"):
            try:
                hints = generate_hints(
                    st.session_state["article"], st.session_state["question"]
                )
                st.session_state["hints"] = hints
            except Exception as e:
                st.error(f"Error generating hints: {e}")
                st.session_state["hints"] = {}

        hints = st.session_state.get("hints", {})
        hints_used = st.session_state["hints_used"]

        # Progressive reveal with expanders
        if hints:
            if st.button("🔍 Reveal Next Hint"):
                if st.session_state["hints_used"] < 3:
                    st.session_state["hints_used"] += 1
                    st.rerun()

            hints_used = st.session_state["hints_used"]

            if hints_used >= 1:
                with st.expander("💡 Hint 1 (Low Detail)", expanded=True):
                    st.write(hints.get("Hint 1 (Low Detail)", ""))

            if hints_used >= 2:
                with st.expander("💡 Hint 2 (Medium Detail)", expanded=True):
                    st.write(hints.get("Hint 2 (Medium Detail)", ""))

            if hints_used >= 3:
                with st.expander("💡 Hint 3 (Near-Explicit)", expanded=True):
                    st.write(hints.get("Hint 3 (Near-Explicit)", ""))

            # Show "Reveal Answer" only after all hints used
            if hints_used >= 3:
                st.markdown("---")
                if st.button("🔓 Reveal Answer"):
                    correct = st.session_state["correct_answer"]
                    opts = st.session_state["options"]
                    st.success(f"The correct answer is **{correct}. {opts.get(correct, '')}**")

            if hints_used < 3:
                st.info(f"Hints revealed: {hints_used}/3. Click 'Reveal Next Hint' to continue.")


# ══════════════════════════════════════════════════════════════
# TAB 4 — Developer / Analytics Dashboard
# ══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Developer Dashboard")
    st.caption("Model performance metrics and session analytics.")

    # --- Model A Metrics ---
    st.markdown("### Model A — Performance")
    metrics_col1, metrics_col2 = st.columns(2)

    with metrics_col1:
        st.markdown("**Available Models:**")
        model_files = {
            "Logistic Regression": "models/model_a/traditional/lr_model.pkl",
            "Linear SVM": "models/model_a/traditional/svm_model.pkl",
            "Naive Bayes": "models/model_a/traditional/nb_model.pkl",
            "Ensemble": "models/model_a/traditional/ensemble_model.pkl",
        }
        for name, path in model_files.items():
            status = "✅ Loaded" if os.path.exists(path) else "❌ Not found"
            st.write(f"- {name}: {status}")

    with metrics_col2:
        st.markdown("**Vectorizers:**")
        vec_files = {
            "One-Hot (Primary)": "models/model_a/traditional/vectorizer.pkl",
            "TF-IDF (Optional)": "models/model_a/traditional/tfidf_vectorizer.pkl",
        }
        for name, path in vec_files.items():
            status = "✅ Loaded" if os.path.exists(path) else "❌ Not found"
            st.write(f"- {name}: {status}")

    # --- Model B Metrics ---
    st.markdown("### Model B — Performance")
    model_b_files = {
        "Word2Vec": "models/model_b/traditional/word2vec.model",
        "Distractor Ranker": "models/model_b/traditional/distractor_ranker.pkl",
        "Hint Scorer": "models/model_b/traditional/hint_scorer.pkl",
    }
    for name, path in model_b_files.items():
        status = "✅ Loaded" if os.path.exists(path) else "❌ Not found"
        st.write(f"- {name}: {status}")

    st.info("Run `python src/evaluate.py` to see full Accuracy, F1, Confusion Matrix, and R² metrics.")

    # --- Session Log ---
    st.markdown("### Session Log")
    log = st.session_state.get("session_log", [])

    if log:
        log_df = pd.DataFrame(log)
        st.dataframe(log_df, use_container_width=True)

        # Latency tracking
        latency_entries = [e for e in log if "latency_s" in e or "question_latency_s" in e]
        if latency_entries:
            st.markdown("**Inference Latency:**")
            for entry in latency_entries:
                if "latency_s" in entry:
                    st.write(f"- {entry['action']}: {entry['latency_s']}s")
                if "question_latency_s" in entry:
                    st.write(f"- Question gen: {entry['question_latency_s']}s, Distractor gen: {entry['distractor_latency_s']}s")

        # CSV Export
        csv_data = session_log_to_csv(log)
        st.download_button(
            label="📥 Export Session Log (CSV)",
            data=csv_data,
            file_name="session_log.csv",
            mime="text/csv",
        )
    else:
        st.write("No actions logged yet. Use the quiz to generate data.")
