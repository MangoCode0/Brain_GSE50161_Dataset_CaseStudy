# Preprocessing
import pandas as pd
import numpy as np
import pickle
import os

# LabelEncoder converts text class names to numbers
# StandardScaler scales all gene values to same range
# train_test_split splits data into train and test
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

#Load the dataset

df = pd.read_csv("data/Brain_GSE50161.csv")
print("\n")
print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
print("\n")

# Separate features (X) and target (y)
print("Separating Features (X) and Target (y):")

# y = target column
# We extract it BEFORE modifying df
y_raw = df['type']   # still text labels like "ependymoma"

# X = everything except 'samples' (ID) and 'type' (label)
X = df.drop(columns=['samples', 'type']) #drop the 'samples' column because it's just an ID number

print(f"X shape (features): {X.shape}")
print(f"y shape (labels)  : {y_raw.shape}")
print(f"Unique classes    : {y_raw.unique()}")

# Remove AFFX control probes
print("\n")
print("Removing AFFX Control Probes:")

before = X.shape[1]

# Keep only columns that do NOT start with "AFFX"
X = X.loc[:, ~X.columns.str.startswith("AFFX")]

after = X.shape[1]
print(f"Columns before removal : {before}")
print(f"AFFX probes removed    : {before - after}")
print(f"Columns after removal  : {after}")
print("These were quality control probes, not real genes")

#Encode text labels into numbers
print("Encoding Labels :")

encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)   # y is now an array of numbers

print("Encoding mapping:")
for number, name in enumerate(encoder.classes_):
    print(f"  {name:25s} → {number}")

print(f"\ny (before): {y_raw.values[:5]}")
print(f"y (after) : {y[:5]}")


# Train/Test Split

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

print(" Scaling Gene Expression Values")

scaler = StandardScaler()

# fit_transform on training data: learn the scale AND apply it
X_train_scaled = scaler.fit_transform(X_train)

# transform only on test data: apply the same scale (no new learning)
X_test_scaled  = scaler.transform(X_test)

print(f"Before scaling — Train mean: {X_train.values.mean():.4f}, std: {X_train.values.std():.4f}")
print(f"After  scaling — Train mean: {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.4f}")
print("After scaling: mean=0, std=1 for all genes — fair comparison")



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
