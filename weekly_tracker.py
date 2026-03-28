import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import date

 #Load weekly tracker data

try:
    df = pd.read_csv('data/weekly_tracker.csv')
except FileNotFoundError:
    print("No trackng data found!")
    print("Run predict.py first to log some entries.")
    exit()

print("=" * 50)
print("WEEKLY FOCUS TRACKER")
print("=" * 50)
print(f"   Total entries logged: {len(df)}")
print("=" * 50)


# Show summary table
print("\n Your Focus Log:")
print("-" * 65)
print(f"{'#':<4} {'Date':<14} {'Study':>6} {'Sleep':>6} "
      f"{'Phone':>6} {'Break':>6} {'Score':>7} {'Level':<10}")
print("-" * 65)

for i, row in df.iterrows():
    print(f"{i+1:<4} {str(row['date']):<14} "
          f"{row['study_hours']:>6.1f} "
          f"{row['sleep_hours']:>6.1f} "
          f"{row['phone_usage']:>6.1f} "
          f"{row['break_time']:>6.1f} "
          f"{row['focus_score']:>7.1f} "
          f"{row['focus_level']:<10}")

print("-" * 65)
print(f"{'Average':<19} "
      f"{df['study_hours'].mean():>6.1f} "
      f"{df['sleep_hours'].mean():>6.1f} "
      f"{df['phone_usage'].mean():>6.1f} "
      f"{df['break_time'].mean():>6.1f} "
      f"{df['focus_score'].mean():>7.1f}")
print("-" * 65)


#Weekly insights

print("\n Your Insights:")
avg_score = df['focus_score'].mean()
best_day  = df.loc[df['focus_score'].idxmax()]
worst_day = df.loc[df['focus_score'].idxmin()]

print(f" Best focus day  : {best_day['date']} "
      f"(Score: {best_day['focus_score']})")
print(f"   Worst focus day : {worst_day['date']} "
      f"(Score: {worst_day['focus_score']})")
print(f"  Average score   : {avg_score:.1f} / 100")

if avg_score > 66:
    print("Overall: You have EXCELLENT focus habits!!")
elif avg_score > 33:
    print(" Overall: Your focus is AVERAGE. Keep improving!!")
else:
    print("Overall: Your focus needs SERIOUS improvement!!")


#Graph 1: Focus score trend over time

if len(df) < 2:
    print("\n Log at least 2 days to see trend graphs.")
    print("   Run predict.py again tomorrow with new values!")
else:
    # Color each point by focus level
    color_map = {'Low': '#FF6B6B', 'Medium': '#FFD93D', 'High': '#6BCB77'}
    point_colors = [color_map.get(level, '#4C9BE8')
                    for level in df['focus_level']]

    plt.figure(figsize=(10, 5))

    # Line connecting all point
    plt.plot(range(len(df)), df['focus_score'],
             color='#4C9BE8', linewidth=2,
             linestyle='--', zorder=1)

    # Color dots for each day
    plt.scatter(range(len(df)), df['focus_score'],
                c=point_colors, s=120,
                edgecolors='black', linewidths=0.5,
                zorder=2)

    # Labels
    for i, score in enumerate(df['focus_score']):
        plt.text(i, score + 1.5, f'{score:.0f}',
                 ha='center', fontsize=10, fontweight='bold')

    # Threshold lines
    plt.axhline(y=33, color='#FF6B6B', linestyle=':',
                linewidth=1.5, label='Low threshold (33)')
    plt.axhline(y=66, color='#6BCB77', linestyle=':',
                linewidth=1.5, label='High threshold (66)')

    plt.xticks(range(len(df)), df['date'], rotation=45, fontsize=9)
    plt.title('Your Daily Focus Score Trend',
              fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Focus Score')
    plt.ylim(0, 100)

    # Legend for colors
    low_patch    = mpatches.Patch(color='#FF6B6B', label='Low')
    medium_patch = mpatches.Patch(color='#FFD93D', label='Medium')
    high_patch   = mpatches.Patch(color='#6BCB77', label='High')
    plt.legend(handles=[low_patch, medium_patch, high_patch],
               title='Focus Level', loc='upper right')

    plt.tight_layout()
    plt.savefig('data/graph8_weekly_trend.png')
    plt.show()
    print("\n Weekly trend graph saved!")

    #Graph 2: Your habits radar
    plt.figure(figsize=(10, 5))

    x = np.arange(len(df))
    width = 0.2

    plt.bar(x - width*1.5, df['study_hours'],
            width, label='Study hrs', color='#4C9BE8',
            edgecolor='black', linewidth=0.5)
    plt.bar(x - width*0.5, df['sleep_hours'],
            width, label='Sleep hrs', color='#6BCB77',
            edgecolor='black', linewidth=0.5)
    plt.bar(x + width*0.5, df['phone_usage'],
            width, label='Phone hrs', color='#FF6B6B',
            edgecolor='black', linewidth=0.5)
    plt.bar(x + width*1.5, df['break_time'] / 10,
            width, label='Break (÷10)', color='#FFD93D',
            edgecolor='black', linewidth=0.5)

    plt.xticks(x, df['date'], rotation=45, fontsize=9)
    plt.title('Your Daily Habits Comparison',
              fontsize=14, fontweight='bold')
    plt.ylabel('Hours')
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/graph9_habits_comparison.png')
    plt.show()
    print("Habits comparison graph saved!")

print("\n" + "=" * 50)
print("   Keep logging daily to see your progress!")
print("=" * 50)
