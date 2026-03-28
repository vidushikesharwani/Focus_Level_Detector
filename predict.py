import pickle
import numpy as np
import pandas as pd
from datetime import date

#Load the saved trained model
# We don't retrain, we just load what we already trained

with open('data/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

print("=" * 50)
print("FOCUS LEVEL DETECTOR")
print("Powered by Machine Learning")
print("=" * 50)
print(f"   Date: {date.today()}")
print("=" * 50)


#Suggestion system
# Based on predicted score, give personalized advice

def get_suggestions(score, study, sleep, phone, brk):
    suggestions = []


    if score < 35:
        level = "Low"
        suggestions.append("Your focus needs serious improvement.")
    elif score < 60:
        level = "Medium"
        suggestions.append("Your focus is average. Small changes will help a lot!")
    else:
        level = "High"
        suggestions.append("Excellent focus! Keep up these great habits!")


#Specific suggestions based on individual inputs
    if sleep < 6:
        suggestions.append(" Sleep more! You're sleeping under 6 hours. Aim for 7-8 hours.")
    elif sleep >= 8:
        suggestions.append(" Great sleep schedule! This is boosting your focus significantly.")

    if phone > 5:
        suggestions.append(" Reduce phone usage! Over 5 hours is seriously hurting your focus.")
    elif phone < 2:
        suggestions.append(" Excellent phone discipline! This is a major focus booster.")

    if study < 3:
        suggestions.append(" Study more! Less than 3 hours is not enough for good focus.")
    elif study >= 6:
        suggestions.append(" Great study hours! Consistency is key — keep it up.")

    if brk < 10:
        suggestions.append(" Take more breaks! Short breaks recharge your brain.")
    elif brk > 45:
        suggestions.append(" Your breaks are too long. Keep them between 15-25 minutes.")
    else:
        suggestions.append(" Good break time! This helps maintain concentration.")

    return level, suggestions

#Take input from user
print("\nEnter your study habits for today:\n")

try:
    study = float(input(" Study hours (e.g. 5.5)     : "))
    sleep = float(input(" Sleep hours (e.g. 7.0)     : "))
    phone = float(input(" Phone usage hours (e.g. 3) : "))
    brk   = float(input(" Break time in mins (e.g. 20): "))
#Predict using the ML model
    input_data = pd.DataFrame([[study, sleep, phone, brk]],
                               columns=['study_hours', 'sleep_hours',
                                        'phone_usage', 'break_time'])

    predicted_score = model.predict(input_data)[0]
    predicted_score = np.clip(predicted_score, 0, 100)

#Get level and suggestions
    level, suggestions = get_suggestions(
        predicted_score, study, sleep, phone, brk
    )

   
    #Display results
   
    print("\n" + "=" * 50)
    print("   YOUR FOCUS REPORT")
    print("=" * 50)
    print(f"   Predicted Focus Score : {predicted_score:.1f} / 100")
    print(f"   Focus Level           : {level}")
    print("=" * 50)
    print("\n Personalized Suggestions:")
    print("-" * 50)
    for i, s in enumerate(suggestions, 1):
        print(f"  {i}. {s}")
    print("-" * 50)

    
#Save this entry to weekly tracker CSV
    
    
    today = str(date.today())

    new_entry = pd.DataFrame([[today, study, sleep, phone,
                                brk, round(predicted_score, 1),
                                level.split()[0]]],
                              columns=['date', 'study_hours', 'sleep_hours',
                                       'phone_usage', 'break_time',
                                       'focus_score', 'focus_level'])

    try:
        tracker = pd.read_csv('data/weekly_tracker.csv')
        tracker = pd.concat([tracker, new_entry], ignore_index=True)
    except FileNotFoundError:
        tracker = new_entry

    tracker.to_csv('data/weekly_tracker.csv', index=False)
    print(f"\n Today's entry saved to weekly_tracker.csv")
    print(f"   Total entries so far: {len(tracker)}")
    print("=" * 50)

except ValueError:
    print("\n Invalid input! Please enter numbers only.")
