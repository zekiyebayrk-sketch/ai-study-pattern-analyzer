"""
generate_data.py
----------------
Creates a synthetic but realistic dataset of study habits.
Run this script first to generate data/study_data.csv.
"""

import numpy as np
import pandas as pd

# Fix the random seed so the dataset is reproducible
np.random.seed(42)

NUM_SAMPLES = 300


def generate_study_data(n: int) -> pd.DataFrame:
    # --- Raw feature generation ---

    sleep_hours = np.random.normal(loc=7.0, scale=1.2, size=n).clip(4, 10)
    study_hours = np.random.normal(loc=4.0, scale=1.5, size=n).clip(0.5, 9)
    screen_time_hours = np.random.normal(loc=4.5, scale=1.8, size=n).clip(0.5, 12)

    # Energy and focus (1–10 scale), mildly correlated with sleep
    energy_level = (sleep_hours / 10 * 7 + np.random.normal(0, 1, n)).clip(1, 10)
    focus_level = (study_hours / 9 * 7 + np.random.normal(0, 1.2, n)).clip(1, 10)

    # Stress tends to rise with screen time and fall with sleep
    stress_level = (
        screen_time_hours / 12 * 8
        - sleep_hours / 10 * 3
        + np.random.normal(4, 1, n)
    ).clip(1, 10)

    # --- Productivity score (1–10) ---
    # Formula reflects real-world intuition:
    #   more sleep, study, energy, and focus → higher score
    #   more screen time and stress → lower score
    raw_score = (
        0.25 * sleep_hours
        + 0.40 * study_hours
        + 0.20 * energy_level
        + 0.25 * focus_level
        - 0.15 * screen_time_hours
        - 0.20 * stress_level
        + np.random.normal(0, 0.4, n)   # small noise so it's not perfectly linear
    )

    # Rescale to the 1–10 range
    min_s, max_s = raw_score.min(), raw_score.max()
    productivity_score = ((raw_score - min_s) / (max_s - min_s) * 9 + 1).round(2)

    df = pd.DataFrame({
        "sleep_hours":       sleep_hours.round(1),
        "study_hours":       study_hours.round(1),
        "screen_time_hours": screen_time_hours.round(1),
        "energy_level":      energy_level.round(1),
        "stress_level":      stress_level.round(1),
        "focus_level":       focus_level.round(1),
        "productivity_score": productivity_score,
    })

    return df


if __name__ == "__main__":
    df = generate_study_data(NUM_SAMPLES)
    output_path = "data/study_data.csv"
    df.to_csv(output_path, index=False)

    print(f"Dataset saved to {output_path}")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nBasic statistics:")
    print(df.describe().round(2))
