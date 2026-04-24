# Preprocessing
# Goal: Prepare the raw data so our ML model can learn from it
# Steps we do here:
#   1. Load data
#   2. Separate features (X) and target label (y)
#   3. Encode text labels → numbers
#   4. Remove control probes (AFFX)
#   5. Scale gene expression values
#   6. Split into train and test sets
#   7. Save everything for next stage

import pandas as pd
import numpy as np
import pickle
import os

# LabelEncoder- converts text class names to numbers
# StandardScaler- scales all gene values to same range
# train_test_split- splits data into train and test
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

#Load the dataset
# Same as before — load CSV into a DataFrame

df = pd.read_csv("data/Brain_GSE50161.csv")
print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")


# Separate features (X) and target (y)
print("Separating Features (X) and Target (y)")

# y = target column
# We extract it BEFORE modifying df
y_raw = df['type']   # still text labels like "ependymoma"

# X = everything except 'samples' (ID) and 'type' (label)
X = df.drop(columns=['samples', 'type']) #drop the 'samples' column because it's just an ID number

print(f"X shape (features): {X.shape}")
print(f"y shape (labels)  : {y_raw.shape}")
print(f"Unique classes    : {y_raw.unique()}")

# Remove AFFX control probes
# Columns starting with "AFFX-" are technical control probes
# used by the Affymetrix chip for quality checking.
# They are NOT real genes — including them adds noise.
# We filter them out by keeping only columns that do NOT start with AFFX
print("Removing AFFX Control Probes")

before = X.shape[1]

# Keep only columns that do NOT start with "AFFX"
# ~ means NOT in pandas
# str.startswith("AFFX") returns True for control probe columns
X = X.loc[:, ~X.columns.str.startswith("AFFX")]

after = X.shape[1]
print(f"Columns before removal : {before}")
print(f"AFFX probes removed    : {before - after}")
print(f"Columns after removal  : {after}")
print("These were quality control probes, not real genes")

#Encode text labels into numbers
# ML models only work with numbers, not text.
# LabelEncoder assigns a number to each unique class alphabetically

# fit_transform() = learn the mapping AND apply it in one step
# We also SAVE the encoder so later we can reverse:
print("Encoding Labels :")

encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)   # y is now an array of numbers

print("Encoding mapping:")
for number, name in enumerate(encoder.classes_):
    print(f"  {name:25s} → {number}")

print(f"\ny (before): {y_raw.values[:5]}")
print(f"y (after) : {y[:5]}")


# Train/Test Split
#
# We split BEFORE scaling. Why?
# Because the scaler must learn (fit) only from training data.
# If we scale the full dataset first, the test data's information
# "leaks" into the scaler — that's called data leakage.
# The model would be tested on data it indirectly "saw" during scaling.
#
# stratify=y ensures each tumor type is proportionally represented
# in both train and test sets.
# random_state=42 makes the split reproducible (same result every run)
print("STEP 5: Train/Test Split (80% train, 20% test)")

X_train, X_test, y_train, y_test = train_test_split(
    X,              # features
    y,              # labels (encoded)
    test_size=0.2,  # 20% goes to test set
    random_state=42,# makes the split same every time you run
    stratify=y      # ensures balanced class distribution in both sets
)

print(f"Training samples : {X_train.shape[0]}")
print(f"Testing samples  : {X_test.shape[0]}")
print(f"Features (genes) : {X_train.shape[1]}")

# Verify stratification worked
print("\nClass distribution check:")
train_dist = pd.Series(y_train).value_counts().sort_index()
test_dist  = pd.Series(y_test).value_counts().sort_index()

for i, name in enumerate(encoder.classes_):
    print(f"  {name:25s} → train: {train_dist.get(i,0):3d} | test: {test_dist.get(i,0):3d}")


#Feature Scaling

# StandardScaler transforms each gene column so that:
#   mean = 0
#   standard deviation = 1
#
# Formula: scaled_value = (original_value - mean) / std
#
# CRITICAL RULE:
#   scaler.fit_transform(X_train) → learns mean/std FROM train, applies it
#   scaler.transform(X_test)      → applies SAME mean/std to test (no learning)
#
# We never fit on test data — that would be cheating.

print("\n" + "=" * 55)
print("STEP 6: Scaling Gene Expression Values")
print("=" * 55)

scaler = StandardScaler()

# fit_transform on training data: learn the scale AND apply it
X_train_scaled = scaler.fit_transform(X_train)

# transform only on test data: apply the same scale (no new learning)
X_test_scaled  = scaler.transform(X_test)

print(f"Before scaling — Train mean: {X_train.values.mean():.4f}, std: {X_train.values.std():.4f}")
print(f"After  scaling — Train mean: {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.4f}")
print("After scaling: mean≈0, std≈1 for all genes — fair comparison")


#  Save everything
#
# We save:
#   X_train_scaled, X_test_scaled  → preprocessed feature matrices
#   y_train, y_test                → encoded labels
#   encoder                        → to convert numbers back to names
#   scaler                         → to scale new data in the app
#   feature_names                  → gene names (needed for biomarker stage)
#
# pickle.dump() serializes Python objects to binary files
# This lets us load them in the next scripts without reprocessing
print(" Saving Preprocessed Data:")

os.makedirs("model", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Save arrays as numpy files (.npy)
np.save("data/processed/X_train.npy", X_train_scaled)
np.save("data/processed/X_test.npy",  X_test_scaled)
np.save("data/processed/y_train.npy", y_train)
np.save("data/processed/y_test.npy",  y_test)

# Save encoder and scaler as pickle files
pickle.dump(encoder, open("model/encoder.pkl", "wb"))
pickle.dump(scaler,  open("model/scaler.pkl",  "wb"))

# Save gene names — we'll need these for biomarker discovery later
feature_names = list(X.columns)
pickle.dump(feature_names, open("model/feature_names.pkl", "wb"))

print("Saved: data/processed/X_train.npy")
print("Saved: data/processed/X_test.npy")
print("Saved: data/processed/y_train.npy")
print("Saved: data/processed/y_test.npy")
print("Saved: model/encoder.pkl")
print("Saved: model/scaler.pkl")
print("Saved: model/feature_names.pkl")

print("\nPreprocessing complete")
print(f"   {X_train_scaled.shape[0]} training samples ready")
print(f"   {X_test_scaled.shape[0]}  testing samples ready")
print(f"   {X_train_scaled.shape[1]} gene features")
