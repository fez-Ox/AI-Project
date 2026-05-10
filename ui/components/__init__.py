"""Reusable helper functions for the Streamlit UI."""

import os
import time

import pandas as pd


def load_random_sample():
    """Load a random row from the RACE dataset for quick testing."""
    raw_path = "data/raw/train.csv"
    if not os.path.exists(raw_path):
        return None

    df = pd.read_csv(raw_path)
    row = df.sample(1).iloc[0]
    return {
        "article": str(row["article"]),
        "question": str(row["question"]),
        "options": {
            "A": str(row["A"]),
            "B": str(row["B"]),
            "C": str(row["C"]),
            "D": str(row["D"]),
        },
        "answer": str(row["answer"]).strip().upper(),
    }


def measure_latency(func, *args, **kwargs):
    """Run a function and return (result, elapsed_seconds)."""
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    return result, elapsed


def append_session_log(log_list, entry):
    """Append an entry dict to the session log list."""
    log_list.append(entry)
    return log_list


def session_log_to_csv(log_list):
    """Convert session log to CSV string for download."""
    if not log_list:
        return ""
    df = pd.DataFrame(log_list)
    return df.to_csv(index=False)
