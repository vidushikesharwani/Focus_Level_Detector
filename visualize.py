import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load the dataset
df = pd.read_csv('data/student_focus_data.csv')

print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")

sns.set_style("whitegrid")

# GRAPH 1 - Bar Chart: Focus Level Distribution
# Most important — shows your data is balanced
plt.figure(figsize=(7, 5))

counts = df['focus_level'].value_counts()
colors = ['#FF6B6B', '#FFD93D', '#6BCB77']
bars = plt.bar(counts.index, counts.values,
               color=colors, edgecolor='black', width=0.5)

for bar, count in zip(bars, counts.values):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 1,
             str(count),
             ha='center', va='bottom',
             fontsize=12, fontweight='bold')

plt.title('Focus Level Distribution of Students',
          fontsize=14, fontweight='bold')
plt.xlabel('Focus Level', fontsize=12)
plt.ylabel('Number of Students', fontsize=12)
plt.tight_layout()
plt.savefig('data/graph1_focus_distribution.png')
plt.show()
print("Graph 1 saved!")


# GRAPH 2 - Heatmap: Correlation between all features
# Most important — shows which habit matters most
plt.figure(figsize=(7, 5))

numeric_df = df[['study_hours', 'sleep_hours',
                  'phone_usage', 'break_time', 'focus_score']]
correlation = numeric_df.corr()

sns.heatmap(correlation,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            linewidths=0.5,
            square=True)

plt.title('Feature Correlation Heatmap',
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('data/graph2_heatmap.png')
plt.show()
print("Graph 2 saved!")

print("\n" + " =" * 45)
print("   Both graphs created and saved!")
print("=" * 45)