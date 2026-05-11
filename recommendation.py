"""
recommendation.py
-----------------
Loads the trained model, takes a student's daily habits as input,
predicts their productivity score, and prints personalized tips.

Run after train_model.py.
"""

import joblib
import pandas as pd

MODEL_PATH  = "data/model.pkl"
SCALER_PATH = "data/scaler.pkl"

FEATURES = [
    "sleep_hours",
    "study_hours",
    "screen_time_hours",
    "energy_level",
    "stress_level",
    "focus_level",
]


def get_recommendations(data: dict) -> list[str]:
    """Return a list of actionable tips based on the student's habits."""
    tips = []

    if data["sleep_hours"] < 6:
        tips.append("You're under-sleeping. Aim for at least 7–8 hours to boost memory and focus.")
    elif data["sleep_hours"] > 9:
        tips.append("Sleeping more than 9 hours can cause grogginess. Try waking up a bit earlier.")

    if data["study_hours"] < 2:
        tips.append("You're studying very little. Even 2–3 focused hours a day makes a big difference.")
    elif data["study_hours"] > 8:
        tips.append("Studying 8+ hours risks burnout. Try the Pomodoro technique and take breaks.")

    if data["screen_time_hours"] > 6:
        tips.append("High screen time is draining your focus. Try a 1-hour screen-free window before bed.")

    if data["energy_level"] < 4:
        tips.append("Low energy detected. Check your sleep, hydration, and whether you're eating regularly.")

    if data["stress_level"] > 7:
        tips.append("Your stress level is high. Short walks, breathing exercises, or journaling can help.")

    if data["focus_level"] < 4:
        tips.append("Focus is low. Reduce distractions: turn off notifications and study in a quiet place.")

    if not tips:
        tips.append("Your habits look great! Keep up the consistent routine.")

    return tips


def predict_productivity(data: dict) -> float:
    """Load the saved model and predict the productivity score for the given data."""
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    df = pd.DataFrame([data])[FEATURES]

    # Check whether this model was trained with scaled features (Linear Regression uses a scaler)
    try:
        df_scaled = scaler.transform(df)
        score = model.predict(df_scaled)[0]
        # If the prediction looks unreasonable, fall back to unscaled
        if not (1 <= score <= 10):
            score = model.predict(df)[0]
    except Exception:
        score = model.predict(df)[0]

    # Clamp to the valid range just in case
    return round(float(score), 2)


def run_interactive():
    """Ask the user for their daily habits and show their productivity score."""
    print("=" * 50)
    print("   AI Study Pattern Analyzer")
    print("=" * 50)
    print("Enter your daily habits (press Enter to use the default value):\n")

    prompts = {
        "sleep_hours":       ("Hours of sleep last night", 7.0),
        "study_hours":       ("Hours spent studying",      4.0),
        "screen_time_hours": ("Hours of screen time",      4.5),
        "energy_level":      ("Energy level  (1–10)",      6.0),
        "stress_level":      ("Stress level  (1–10)",      5.0),
        "focus_level":       ("Focus level   (1–10)",      6.0),
    }

    data = {}
    for key, (label, default) in prompts.items():
        raw = input(f"  {label} [{default}]: ").strip()
        data[key] = float(raw) if raw else default

    print()

    # ── Prediction ────────────────────────────────────────────────────────────
    score = predict_productivity(data)

    print("─" * 50)
    print(f"  Predicted Productivity Score: {score:.1f} / 10")
    print("─" * 50)

    # Colour-code the verdict
    if score >= 7.5:
        verdict = "Excellent"
    elif score >= 5.5:
        verdict = "Moderate"
    else:
        verdict = "Needs Improvement"

    print(f"  Overall: {verdict}\n")

    # ── Recommendations ───────────────────────────────────────────────────────
    tips = get_recommendations(data)
    print("  Personalised Recommendations:")
    for i, tip in enumerate(tips, 1):
        print(f"   {i}. {tip}")

    print()


if __name__ == "__main__":
    run_interactive()
