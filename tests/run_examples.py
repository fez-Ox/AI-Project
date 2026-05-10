import os
import sys

# Add the src directory to the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference import (
    generate_distractors,
    generate_hints,
    generate_question,
    predict_answer,
)


def run_example():
    # Example extracted from typical reading comprehension tasks
    article = (
        "The Apollo 11 mission was the first manned mission to land on the Moon. "
        "It was launched by NASA on July 16, 1969. "
        "Astronauts Neil Armstrong and Buzz Aldrin were the first two humans to walk on the lunar surface. "
        "Michael Collins flew the command module Columbia alone in lunar orbit while they were on the Moon's surface."
    )

    question = "Who flew the command module Columbia?"

    options = {
        "A": "Neil Armstrong",
        "B": "Buzz Aldrin",
        "C": "Michael Collins",
        "D": "Yuri Gagarin",
    }

    correct_answer = "Michael Collins"

    print("=" * 60)
    print(" READING COMPREHENSION SYSTEM - TEST EXAMPLES")
    print("=" * 60)
    print(f"[ARTICLE]:\n{article}\n")
    print(f"[QUESTION]: {question}\n")

    print("-" * 60)
    print("1. MODEL A - QUESTION/ANSWER VERIFICATION (predict_answer)")
    print("-" * 60)
    try:
        prediction = predict_answer(article, question, options)
        print(f"Options provided: {options}")
        print(
            f"Predicted Option: {prediction['predicted_option']} - {prediction['predicted_text']}"
        )
        print(
            f"Is it correct? {'YES' if prediction['predicted_text'] == correct_answer else 'NO'}"
        )
        print("\nConfidence Scores:")
        for k, v in prediction["confidence_scores"].items():
            print(f"  {k}: {v:.4f}")
    except FileNotFoundError as e:
        print(
            f"Error: {e}\n(Please ensure you ran 'python src/model_a_train.py' first)"
        )

    print("\n" + "-" * 60)
    print("2. MODEL A - QUESTION GENERATION (generate_question)")
    print("-" * 60)
    generated_q = generate_question(article)
    print(f"Generated Question: {generated_q}")

    print("\n" + "-" * 60)
    print("3. MODEL B - DISTRACTOR GENERATION (generate_distractors)")
    print("-" * 60)
    try:
        print(f"Correct Answer: {correct_answer}")
        distractors = generate_distractors(article, question, correct_answer)
        print("Generated Distractors:")
        for i, dist in enumerate(distractors):
            print(f"  {i + 1}. {dist}")
    except FileNotFoundError as e:
        print(
            f"Error: {e}\n(Please ensure you ran 'python src/model_b_train.py' first)"
        )

    print("\n" + "-" * 60)
    print("4. MODEL B - HINT GENERATION (generate_hints)")
    print("-" * 60)
    hints = generate_hints(article, question)
    for level, text in hints.items():
        print(f"[{level}]: {text}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_example()
