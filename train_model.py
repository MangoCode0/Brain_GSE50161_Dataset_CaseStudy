# PCA,ModelTraining,Evaluation
import matplotlib
matplotlib.use('Agg') 
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

os.makedirs("outputs", exist_ok=True)
os.makedirs("model",   exist_ok=True)

# Load preprocessed data
print("Loading Preprocessed Data:")
print('-'*40)
X_train = np.load("data/processed/X_train.npy")
X_test  = np.load("data/processed/X_test.npy")
y_train = np.load("data/processed/y_train.npy")
y_test  = np.load("data/processed/y_test.npy")

encoder = pickle.load(open("model/encoder.pkl", "rb"))
print(f"X_train shape : {X_train.shape}")  # (104, 54613)
print(f"X_test  shape : {X_test.shape}")   # (26,  54613)
print(f"Classes  : \n{list(encoder.classes_)}")

# We fit PCA on train data only 
print("\nFinding Optimal PCA Components:")
print('-'*40)
# First fit PCA with all possible components to see variance curve
pca_full = PCA(random_state=42)
pca_full.fit(X_train)  # fit on train only — learns the structure

# Cumulative explained variance
# cumsum adds up values one by one
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Find how many components reach 90% variance
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1  # +1 because index starts at 0
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

print(f"Components needed for 90% variance: {n_components_90}")
print(f"Components needed for 95% variance: {n_components_95}")

# We'll use 50 components 
# covers majority of variance, keeps model fast
N_COMPONENTS = 50
print(f"\nChosen: {N_COMPONENTS} components")
print(f"Variance explained by {N_COMPONENTS} components: {cumulative_variance[N_COMPONENTS-1]*100:.2f}%")

# Plot the variance curve — helps visualize the "elbow"
plt.figure(figsize=(9, 4))
plt.plot(range(1, 101), cumulative_variance[:100] * 100,
         color='#6d28d9', linewidth=2)

# Vertical line at our chosen n_components
plt.axvline(x=N_COMPONENTS, color='red', linestyle='--',
            label=f'Chosen: {N_COMPONENTS} components')

# Horizontal line at 90%
plt.axhline(y=90, color='orange', linestyle='--', label='90% variance')

plt.xlabel("Number of Principal Components", fontsize=12)
plt.ylabel("Cumulative Explained Variance (%)", fontsize=12)
plt.title("PCA: How Many Components Do We Need?", fontsize=13, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/pca_variance_curve.png", dpi=150)
# plt.show()
print("Chart saved: outputs/pca_variance_curve.png")

#Apply PCA
# Now we actually reduce dimensions:
print("\nApplying PCA:")
print('-'*40)
pca = PCA(n_components=N_COMPONENTS, random_state=42)

X_train_pca = pca.fit_transform(X_train)  # learn + apply on train
X_test_pca  = pca.transform(X_test)       # apply only on test

print(f"X_train: {X_train.shape} → {X_train_pca.shape}")
print(f"X_test : {X_test.shape}  → {X_test_pca.shape}")
print(f"Dimensionality reduced by: {round((1 - N_COMPONENTS/X_train.shape[1])*100, 2)}%")

# Visualize first 2 PCA components (PC1 vs PC2)
# This is a 2D "map" of where each sample sits in PCA space
# If classes cluster separately, the model will learn easily
plt.figure(figsize=(8, 6))
colors = ['#7c3aed', '#2563eb', '#059669', '#d97706', '#dc2626']
class_names = list(encoder.classes_)

for i, (name, color) in enumerate(zip(class_names, colors)):
    # Select only the samples belonging to class i
    mask = y_train == i
    plt.scatter(
        X_train_pca[mask, 0],   # PC1 values for this class
        X_train_pca[mask, 1],   # PC2 values for this class
        label=name,
        color=color,
        alpha=0.7,
        s=60
    )

plt.xlabel("Principal Component 1", fontsize=12)
plt.ylabel("Principal Component 2", fontsize=12)
plt.title("PCA: Tumor Types in 2D Space\n(PC1 vs PC2)", fontsize=13, fontweight='bold')
plt.legend(fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/pca_2d_scatter.png", dpi=150)
print("Chart saved: outputs/pca_2d_scatter.png")

# Train the Random Forest
print('-'*40)
print("Random forest Training with 200 trees")
print('-'*40)
model = RandomForestClassifier(
    n_estimators=200,       # number of trees
    max_depth=10,            # max depth per tree
    class_weight='balanced',   # handles our imbalanced classes
    random_state=42,        # reproducibility
    n_jobs=-1                   # use all CPU cores
)

model.fit(X_train_pca, y_train)
print("Training complete!")

# Cross Validation
print("\nStratified 5-Fold Cross Validation")
print('-'*40)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# cross_val_score runs the model 5 times on different splits
# and returns an array of 5 scores
cv_scores = cross_val_score(
    model,          # our model
    X_train_pca,    # features
    y_train,        # labels
    cv=cv,          # cross validation strategy
    scoring='f1_macro'  # metric
)

print(f"F1 Score per fold : \n{[round(s, 4) for s in cv_scores]}")
print(f"Mean F1 Score     : {cv_scores.mean():.4f}")
print(f"Std Deviation     : {cv_scores.std():.4f}")
print("Low std = model is consistent across different data splits")

#Final Evaluation on Test Set
# confusion_matrix shows which classes were confused with which
print("\nFinal Evaluation on Test Set")
print('-'*40)
y_pred = model.predict(X_test_pca)

# Overall accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc*100:.2f}%")

# Detailed per-class report
print("\nDetailed Classification Report:")
print('-'*40)
print(classification_report(
    y_test,
    y_pred,
    target_names=encoder.classes_
))

# Explanation of report columns:
print("\nColumn meanings:")
print('-'*40)
print("  precision : when model predicts X, how often is it correct?")
print("  recall    : of all actual X cases, how many did model find?")
print("  f1-score  : harmonic mean of precision & recall")
print("  support   : how many samples of this class in test set")

# Confusion Matrix
print("\nConfusion Matrix:")
print('-'*40)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,          # show numbers in each cell
    fmt='d',             # format as integer
    cmap='Purples',      # color scheme
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)
plt.title("Confusion Matrix — Test Set", fontsize=13, fontweight='bold')
plt.ylabel("Actual Tumor Type", fontsize=11)
plt.xlabel("Predicted Tumor Type", fontsize=11)
plt.xticks(rotation=20, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)
# plt.show()
print("Chart saved: outputs/confusion_matrix.png")

#Save the trained model and PCA
print("\nSaving Model and PCA:")
print('-'*40)
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(pca,   open("model/pca.pkl",   "wb"))
print("Saved: model/model.pkl")
print("Saved: model/pca.pkl")

print("\nTRAINING COMPLETE ")
print('-'*40)
print(f"  Features used       : {N_COMPONENTS} PCA components (from 54,613 genes)")
print(f"  Training samples    : {X_train_pca.shape[0]}")
print(f"  Test samples        : {X_test_pca.shape[0]}")
print(f"  Cross-val F1 (mean) : {cv_scores.mean():.4f}")
print(f"  Test Accuracy       : {acc*100:.2f}%")
