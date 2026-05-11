# AI Study Pattern Analyzer

A beginner-friendly machine learning project that analyzes daily study habits and predicts a **productivity score from 1 to 10**.

---

## What it does

Given six features about a student's day, a regression model estimates how productive that day is likely to be:

| Feature | Description |
|---|---|
| `sleep_hours` | Hours of sleep the previous night |
| `study_hours` | Hours spent actively studying |
| `screen_time_hours` | Total hours on screens (phone, social media, etc.) |
| `energy_level` | Self-reported energy (1 = exhausted, 10 = very energetic) |
| `stress_level` | Self-reported stress (1 = calm, 10 = very stressed) |
| `focus_level` | Self-reported focus quality (1 = distracted, 10 = laser-focused) |

---

## Project structure

```
ai-study-pattern-analyzer/
├── data/
│   └── study_data.csv          # Generated dataset (300 rows)
├── src/
│   ├── generate_data.py        # Step 1 — create synthetic dataset
│   ├── train_model.py          # Step 2 — train & evaluate the model
│   └── recommendation.py      # Step 3 — interactive predictor + tips
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate the dataset

```bash
python src/generate_data.py
```

This creates `data/study_data.csv` with 300 synthetic but realistic rows.

### 3. Train the model

```bash
python src/train_model.py
```

This will:
- Train a **Linear Regression** and a **Random Forest** model
- Print MAE and R² scores for both
- Save the best model to `data/model.pkl`
- Save two plots: `actual_vs_predicted.png` and `feature_importance.png`

### 4. Get your productivity score

```bash
python src/recommendation.py
```

You'll be prompted to enter your daily habits. The model will predict your score and give personalised recommendations.

---

## Example output

```
Enter your daily habits:

  Hours of sleep last night [7.0]: 6
  Hours spent studying [4.0]: 5
  Hours of screen time [4.5]: 7
  Energy level  (1–10) [6.0]: 5
  Stress level  (1–10) [5.0]: 7
  Focus level   (1–10) [6.0]: 5

──────────────────────────────────────────────────
  Predicted Productivity Score: 5.3 / 10
──────────────────────────────────────────────────
  Overall: Moderate

  Personalised Recommendations:
   1. You're under-sleeping. Aim for at least 7–8 hours.
   2. High screen time is draining your focus. Try a 1-hour screen-free window before bed.
   3. Your stress level is high. Short walks or breathing exercises can help.
```

---

## How the model works

1. **Data generation** — features are drawn from realistic normal distributions. The productivity score is a weighted combination of all features plus small random noise.
2. **Preprocessing** — features are standardised with `StandardScaler` (zero mean, unit variance).
3. **Models** — `LinearRegression` and `RandomForestRegressor` are both trained; the one with the lower Mean Absolute Error is saved.
4. **Evaluation** — MAE measures average prediction error in score units; R² measures how much variance the model explains.

---

## Tech stack

- **pandas** — data loading and manipulation
- **numpy** — numerical operations and data generation
- **scikit-learn** — model training, scaling, and evaluation
- **matplotlib** — visualisation
- **joblib** — saving and loading the trained model
