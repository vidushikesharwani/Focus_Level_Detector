# ============================================================
# FILE: generate.py
# PURPOSE: Generate a realistic student focus dataset
# PROJECT: Focus Level Detector
# ============================================================

import pandas as pd
import numpy as np

np.random.seed(10)  # changed seed for better distribution

n = 200

# --- Generate features in 3 groups for realistic spread ---

# Group 1 → Low focus students (70 students)
# Bad habits: low sleep, low study, high phone
n_low = 70
study_low  = np.random.uniform(1, 4, n_low)
sleep_low  = np.random.uniform(4, 6, n_low)
phone_low  = np.random.uniform(5, 8, n_low)
break_low  = np.random.uniform(5, 15, n_low)

# Group 2 → Medium focus students (80 students)
# Average habits
n_med = 80
study_med  = np.random.uniform(3, 7, n_med)
sleep_med  = np.random.uniform(6, 8, n_med)
phone_med  = np.random.uniform(3, 6, n_med)
break_med  = np.random.uniform(15, 35, n_med)

# Group 3 → High focus students (50 students)
# Good habits: high sleep, high study, low phone
n_high = 50
study_high = np.random.uniform(6, 10, n_high)
sleep_high = np.random.uniform(7, 10, n_high)
phone_high = np.random.uniform(0, 3, n_high)
break_high = np.random.uniform(20, 40, n_high)

# --- Combine all groups ---
study_hours = np.concatenate([study_low, study_med, study_high])
sleep_hours = np.concatenate([sleep_low, sleep_med, sleep_high])
phone_usage = np.concatenate([phone_low, phone_med, phone_high])
break_time  = np.concatenate([break_low, break_med, break_high])

# --- Focus score formula ---

# Sleep → max 40 points
sleep_score = (sleep_hours / 10) * 40

# Study → max 30 points
study_score = (study_hours / 10) * 30

# Break → max 15 points (optimal = 20 mins)
ideal_break = 20
break_score = 15 - abs(break_time - ideal_break) * 0.25
break_score = np.clip(break_score, 0, 15)

# Phone → penalty up to 25 points
phone_penalty = (phone_usage / 8) * 25

# Combine
raw_score = sleep_score + study_score + break_score - phone_penalty

# Small realistic noise
noise = np.random.normal(0, 2, n)
raw_score += noise

# Clamp to 0-100
focus_score = np.clip(raw_score, 0, 100).round(2)

# --- Classification ---
def classify_focus(score):
    if score <= 33:
        return "Low"
    elif score <= 66:
        return "Medium"
    else:
        return "High"

focus_level = [classify_focus(s) for s in focus_score]

# --- Shuffle so groups are mixed randomly ---
df = pd.DataFrame({
    'study_hours': study_hours.round(2),
    'sleep_hours': sleep_hours.round(2),
    'phone_usage': phone_usage.round(2),
    'break_time' : break_time.round(2),
    'focus_score': focus_score,
    'focus_level': focus_level
})

# Shuffle rows so Low/Medium/High are mixed
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
df.to_csv('data/student_focus_data.csv', index=False)

# --- Print results ---
print("=" * 45)
print("   Dataset created successfully!")
print("=" * 45)
print(f"Total records : {len(df)}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nFocus Level Distribution:")
print(df['focus_level'].value_counts())
print(f"\nAverage focus score : {df['focus_score'].mean():.2f}")
print(f"Lowest score        : {df['focus_score'].min():.2f}")
print(f"Highest score       : {df['focus_score'].max():.2f}")
print("=" * 45)