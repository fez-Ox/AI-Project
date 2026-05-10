import os
import sys

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

st.set_page_config(page_title="AI Reading Comprehension", layout="wide")

st.title("📚 Intelligent Reading Comprehension System")
st.markdown(
    "Upload a passage, or paste one below, to generate questions, distractors, hints, and verify answers!"
)

# --- Sidebar Controls ---
st.sidebar.header("Settings")
use_ensemble = st.sidebar.checkbox("Use Ensemble Model (Model A)", value=True)

# --- Main Layout ---
article_input = st.text_area(
    "Enter Reading Passage (Article):",
    height=200,
    placeholder="e.g. The Apollo 11 mission was the first manned mission to land on the Moon...",
)

if article_input:
    st.success("Passage loaded successfully.")

    st.markdown("---")

    col1, col2 = st.columns(2)

    # ---------------------------------------------------------
    # LEFT COLUMN: Question Generation & Distractors (Model A/B)
    # ---------------------------------------------------------
    with col1:
        st.subheader("1. Generate a Quiz")
        if st.button("Generate Question from Passage"):
            with st.spinner("Generating..."):
                gen_q = generate_question(article_input)
                st.session_state["generated_question"] = gen_q
                st.info(f"**Generated Question:** {gen_q}")

        if "generated_question" in st.session_state:
            st.write("---")
            st.write(f"**Current Question:** {st.session_state['generated_question']}")
            correct_ans_input = st.text_input(
                "Enter the correct answer for this question to generate distractors:"
            )

            if correct_ans_input and st.button("Generate Distractors (Wrong Options)"):
                with st.spinner("Finding semantic distractors..."):
                    try:
                        distractors = generate_distractors(
                            article_input,
                            st.session_state["generated_question"],
                            correct_ans_input,
                        )
                        st.write("**Generated Plausible Distractors:**")
                        for i, d in enumerate(distractors):
                            st.error(f"{chr(65 + i)}. {d}")
                    except Exception as e:
                        st.error(f"Error generating distractors: {e}")

            if st.button("Generate Graduated Hints"):
                with st.spinner("Ranking sentences..."):
                    hints = generate_hints(
                        article_input, st.session_state["generated_question"]
                    )
                    st.write("**Hints:**")
                    st.warning(
                        f"**Hint 1 (Low Detail):** {hints['Hint 1 (Low Detail)']}"
                    )
                    st.warning(
                        f"**Hint 2 (Medium Detail):** {hints['Hint 2 (Medium Detail)']}"
                    )
                    st.success(
                        f"**Hint 3 (Near Explicit):** {hints['Hint 3 (Near-Explicit)']}"
                    )

    # ---------------------------------------------------------
    # RIGHT COLUMN: Answer Verification (Model A)
    # ---------------------------------------------------------
    with col2:
        st.subheader("2. Verify an Answer")
        st.markdown("Test the AI's ability to solve a multiple-choice question.")

        user_q = st.text_input("Enter a Question:")
        opt_a = st.text_input("Option A:")
        opt_b = st.text_input("Option B:")
        opt_c = st.text_input("Option C:")
        opt_d = st.text_input("Option D:")

        if st.button("Predict Answer"):
            if not user_q or not opt_a or not opt_b or not opt_c or not opt_d:
                st.error("Please fill in the question and all 4 options.")
            else:
                options = {"A": opt_a, "B": opt_b, "C": opt_c, "D": opt_d}
                with st.spinner("Analyzing text overlap and semantics..."):
                    try:
                        prediction = predict_answer(
                            article_input, user_q, options, use_ensemble=use_ensemble
                        )

                        st.success(
                            f"**Predicted Answer:** {prediction['predicted_option']} - {prediction['predicted_text']}"
                        )

                        st.write("**Confidence Scores:**")

                        # Create a nice bar chart for confidence
                        scores_df = pd.DataFrame(
                            {
                                "Option": list(prediction["confidence_scores"].keys()),
                                "Confidence": list(
                                    prediction["confidence_scores"].values()
                                ),
                            }
                        )
                        st.bar_chart(scores_df.set_index("Option"))

                    except FileNotFoundError as e:
                        st.error(f"Error: {e}. Did you run model_a_train.py?")
