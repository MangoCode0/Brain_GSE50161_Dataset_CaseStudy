# ============================================================
# STAGE 2: Explore the Data
# Goal: Understand what's inside our dataset before touching it
# ============================================================

# --- IMPORTS ---
# pandas: handles our data table (like Excel in Python)
# matplotlib & seaborn: for drawing charts
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------
# STEP 1: Load the dataset
# pd.read_csv() reads the CSV file and stores it as a DataFrame (df)
# A DataFrame is like a smart table with rows and columns
# ---------------------------------------------------------------
df = pd.read_csv("data/Brain_GSE50161.csv")

# ---------------------------------------------------------------
# STEP 2: Basic shape
# df.shape returns (number_of_rows, number_of_columns)
# [0] = rows (samples/patients), [1] = columns (genes + metadata)
# ---------------------------------------------------------------
print("=" * 50)
print("DATASET SHAPE")
print("=" * 50)
print(f"Total Samples (rows):{df.shape[0]}")
print(f"Total Columns       :{df.shape[1]}")
print(f"Gene columns only   : {df.shape[1] - 2}")  # subtract 'samples' and 'type'

# ---------------------------------------------------------------
# STEP 3: Look at the first few rows
# head(3) shows the first 3 rows so we understand the structure
# Notice: first 2 columns are 'samples' (ID) and 'type' (label)
# Everything else is a gene name with its expression value
# ---------------------------------------------------------------
print("\n" + "=" * 50)
print("FIRST 3 ROWS (first 6 columns only)")
print("=" * 50)
print(df.iloc[:3, :6])  # iloc = select by position: rows 0-2, cols 0-5

# ---------------------------------------------------------------
# STEP 4: Check class distribution
# 'type' column = what kind of tumor (our prediction target)
# value_counts() counts how many samples per class
# ---------------------------------------------------------------
print("\n" + "=" * 50)
print("CLASS DISTRIBUTION (Tumor Types)")
print("=" * 50)
class_counts = df['type'].value_counts()
print(class_counts)
imbalance_ratio = class_counts.max() / class_counts.min()
print(f"\nImbalance Ratio (max/min): {imbalance_ratio:.2f}x")
print("Note: Ratio > 3 means imbalanced — we must handle this carefully")

# ---------------------------------------------------------------
# STEP 5: Check for missing values
# isnull() marks every missing cell as True
# .sum().sum() counts ALL missing values across entire dataset
# ---------------------------------------------------------------
print("\n" + "=" * 50)
print("MISSING VALUES CHECK")
print("=" * 50)
total_missing = df.isnull().sum().sum()
print(f"Total missing values: {total_missing}")
if total_missing == 0:
    print("Great! No missing values — dataset is clean.")

# ---------------------------------------------------------------
# STEP 6: Gene expression value range
# We look at just the gene columns (drop 'samples' and 'type')
# describe() gives us stats: min, max, mean, std for all genes
# We take mean across all genes for a summary view
# ---------------------------------------------------------------
print("\n" + "=" * 50)
print("GENE EXPRESSION STATISTICS")
print("=" * 50)
gene_cols = df.drop(columns=['samples', 'type'])
print(f"Min expression value  : {gene_cols.values.min():.3f}")
print(f"Max expression value  : {gene_cols.values.max():.3f}")
print(f"Mean expression value : {gene_cols.values.mean():.3f}")
print("(Values are log2 scale — normal range is roughly 2 to 15)")

# ---------------------------------------------------------------
# STEP 7: Visualize class distribution as a bar chart
# This gives us a visual feel for how balanced/imbalanced it is
# ---------------------------------------------------------------
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

# Add count numbers on top of each bar
for i, count in enumerate(class_counts.values):
    plt.text(i, count + 0.3, str(count), ha='center', fontweight='bold')

plt.tight_layout()  # auto-adjusts spacing so nothing gets cut off

# Save the chart to our outputs folder
import os
os.makedirs("outputs", exist_ok=True)  # create folder if it doesn't exist
plt.savefig("outputs/class_distribution.png", dpi=150)
plt.show()
print("\nChart saved to outputs/class_distribution.png")

# ---------------------------------------------------------------
# STEP 8: Show mean expression per class
# Group samples by tumor type, take mean of all genes per group
# Then take mean across all genes — gives one number per class
# This tells us: do different tumor types have different overall expression levels?
# ---------------------------------------------------------------
print("\n" + "=" * 50)
print("MEAN GENE EXPRESSION PER TUMOR TYPE")
print("=" * 50)
mean_per_class = df.groupby('type')[gene_cols.columns].mean().mean(axis=1)
print(mean_per_class.round(4))
print("\nNote: Similar means = differences are in specific genes, not overall")

print("\n✅ Exploration complete! Move to 2_preprocess.py next.")