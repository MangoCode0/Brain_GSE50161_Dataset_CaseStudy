# Explore the Data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load the dataset
df = pd.read_csv("data/Brain_GSE50161.csv")

#Basic shape
print("DATASET SHAPE")
print(f"Total Samples (rows):{df.shape[0]}")
print(f"Total Columns       :{df.shape[1]}")
print(f"Gene columns only   : {df.shape[1] - 2}")  # subtract 'samples' and 'type'

# Look at the first few rows
print("FIRST 3 ROWS (first 6 columns only)")
print(df.iloc[:3, :6])  # iloc = select by position: rows 0-2, cols 0-5

# Check class distribution
print("CLASS DISTRIBUTION (Tumor Types)")
class_counts = df['type'].value_counts()
print(class_counts)
imbalance_ratio = class_counts.max() / class_counts.min()
print(f"\nImbalance Ratio (max/min): {imbalance_ratio:.2f}x")
print("Note: Ratio > 3 means imbalanced — we must handle this carefully")

# Check for missing values
print("MISSING VALUES CHECK")
total_missing = df.isnull().sum().sum()
print(f"Total missing values: {total_missing}")
if total_missing == 0:
    print("Great! No missing values — dataset is clean.")

# Gene expression value range
print("GENE EXPRESSION STATISTICS")
gene_cols = df.drop(columns=['samples', 'type'])
print(f"Min expression value  : {gene_cols.values.min():.3f}")
print(f"Max expression value  : {gene_cols.values.max():.3f}")
print(f"Mean expression value : {gene_cols.values.mean():.3f}")
print("(Values are log2 scale — normal range is roughly 2 to 15)")

# Visualize class distribution as a bar chart
plt.figure(figsize=(8, 5))  # figsize sets the chart size in inches

# countplot draws a bar for each class, counting occurrences
sns.countplot(
    data=df,
    x='type',
    palette='viridis',   # color scheme
    order=class_counts.index  # sort bars by count
)

# Labels and title
plt.title("Brain Tumor Class Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Tumor Type", fontsize=12)
plt.ylabel("Number of Samples", fontsize=12)
plt.xticks(rotation=15)  # rotate x labels so they don't overlap


# Save the chart to our outputs folder
import os
os.makedirs("outputs", exist_ok=True)  # create folder if it doesn't exist
plt.savefig("outputs/class_distribution.png", dpi=150)
plt.show()
print("\nChart saved to outputs/class_distribution.png")
