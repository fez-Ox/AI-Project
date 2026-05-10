import os

import pandas as pd


def clean_text(text):
    """Clean and normalize text inputs."""
    if pd.isna(text):
        return ""
    return str(text).strip()


def restructure_data(df):
    """
    Converts multiple-choice format to binary classification format.
    Each question generates 4 rows (1 correct, 3 incorrect).
    """
    records = []
    for _, row in df.iterrows():
        article = clean_text(row.get("article", ""))
        question = clean_text(row.get("question", ""))
        correct_ans = (
            str(row.get("answer", "")).strip().upper()
        )  # Expected: 'A', 'B', 'C', 'D'

        options = {
            "A": clean_text(row.get("A", "")),
            "B": clean_text(row.get("B", "")),
            "C": clean_text(row.get("C", "")),
            "D": clean_text(row.get("D", "")),
        }

        for opt_key, opt_text in options.items():
            # Create the combined text feature as specified in the guide
            combined_text = f"{article} {article} {question} {opt_text}"

            # Label 1 if this option is the correct answer, else 0
            label = 1 if opt_key == correct_ans else 0

            records.append(
                {
                    "id": row.get("id", ""),
                    "article": article,
                    "question": question,
                    "option": opt_text,
                    "combined_text": combined_text,
                    "label": label,
                }
            )

    return pd.DataFrame(records)


def preprocess_pipeline(input_path, output_path):
    """Reads raw data, restructures it, and saves it to the processed folder."""
    if not os.path.exists(input_path):
        print(f"Warning: Raw dataset '{input_path}' not found. Skipping.")
        return

    print(f"Processing {input_path}...")
    df = pd.read_csv(input_path)
    processed_df = restructure_data(df)
    processed_df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")


if __name__ == "__main__":
    # Ensure processed directory exists
    os.makedirs("data/processed", exist_ok=True)

    preprocess_pipeline("data/raw/train.csv", "data/processed/train_binary.csv")
    preprocess_pipeline("data/raw/dev.csv", "data/processed/val_binary.csv")
    preprocess_pipeline("data/raw/test.csv", "data/processed/test_binary.csv")
