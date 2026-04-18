# ============================================================
# STAGE 4: PCA + Model Training + Evaluation
#
# Goal: Reduce 54,613 gene features → 50 components using PCA,
#       train a Random Forest classifier, evaluate properly,
#       and save everything for the app and biomarker stage.
#
# Flow:
#   Load preprocessed data
#   → Find best PCA components
#   → Apply PCA
#   → Train Random Forest
#   → Cross Validation
#   → Evaluate on test set
#   → Plot confusion matrix
#   → Save model + PCA
# ============================================================
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


# ---------------------------------------------------------------
# STEP 1: Load preprocessed data
#
# np.load() loads the .npy files we saved in preprocess stage
# allow_pickle=False is safe default for plain arrays
# ---------------------------------------------------------------
print("=" * 55)
print("STEP 1: Loading Preprocessed Data")
print("=" * 55)

X_train = np.load("data/processed/X_train.npy")
X_test  = np.load("data/processed/X_test.npy")
y_train = np.load("data/processed/y_train.npy")
y_test  = np.load("data/processed/y_test.npy")
encoder = pickle.load(open("model/encoder.pkl", "rb"))

print(f"X_train shape : {X_train.shape}")  # (104, 54613)
print(f"X_test  shape : {X_test.shape}")   # (26,  54613)
print(f"Classes       : {list(encoder.classes_)}")


# ---------------------------------------------------------------
# STEP 2: Find the right number of PCA components
#
# We don't just guess 20 or 50 — we check how much variance
# each component explains.
#
# Think of it like this:
#   Component 1 explains 35% of all variation in the data
#   Component 2 explains 12%
#   Component 3 explains 8%
#   ... and so on
#
# We want to keep enough components to explain at least 90% total.
# That's the "elbow" — after that, adding more components gives
# diminishing returns.
#
# We fit PCA on train data only (never touch test data here)
# ---------------------------------------------------------------
print("\n" + "=" * 55)
print("STEP 2: Finding Optimal PCA Components")
print("=" * 55)

# First fit PCA with all possible components to see variance curve
pca_full = PCA(random_state=42)
pca_full.fit(X_train)  # fit on train only — learns the structure

# Cumulative explained variance
# cumsum() adds up values one by one:
# [0.35, 0.12, 0.08, ...] → [0.35, 0.47, 0.55, ...]
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Find how many components reach 90% variance
# np.argmax finds the FIRST position where condition is True
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1  # +1 because index starts at 0
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

print(f"Components needed for 90% variance: {n_components_90}")
print(f"Components needed for 95% variance: {n_components_95}")

# We'll use 50 components — good balance of speed and information
# (covers majority of variance, keeps model fast)
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
plt.show()
print("Chart saved: outputs/pca_variance_curve.png")


# ---------------------------------------------------------------
# STEP 3: Apply PCA
#
# Now we actually reduce dimensions:
#   X_train: 104 samples × 54,613 genes → 104 samples × 50 components
#   X_test :  26 samples × 54,613 genes →  26 samples × 50 components
#
# IMPORTANT:
#   fit_transform on train → PCA learns the components from train data
#   transform on test      → applies same transformation (no new learning)
# ---------------------------------------------------------------
print("\n" + "=" * 55)
print("STEP 3: Applying PCA")
print("=" * 55)

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
plt.show()
print("Chart saved: outputs/pca_2d_scatter.png")


# ---------------------------------------------------------------
# STEP 4: Train the Random Forest
#
# Parameters explained:
#   n_estimators=200  → 200 decision trees vote together
#                        more trees = more stable (but slower)
#   max_depth=10      → each tree can ask max 10 questions
#                        prevents overfitting (memorizing train data)
#   class_weight='balanced' → automatically handles class imbalance
#                        gives more importance to rare classes (Normal=13)
#   random_state=42   → reproducible results
#   n_jobs=-1         → use all CPU cores for speed
# ---------------------------------------------------------------
print("\n" + "=" * 55)
print("STEP 4: Training Random Forest")
print("=" * 55)
print("Training with 200 trees... (this may take 10-30 seconds)")

model = RandomForestClassifier(
    n_estimators=200,       # number of trees
    max_depth=10,           # max depth per tree
    class_weight='balanced',# handles our imbalanced classes
    random_state=42,        # reproducibility
    n_jobs=-1               # use all CPU cores
)

model.fit(X_train_pca, y_train)
print("Training complete!")


# ---------------------------------------------------------------
# STEP 5: Cross Validation
#
# Instead of trusting ONE train/test split, we test the model
# 5 different ways and average the results.
#
# StratifiedKFold = each fold has proportional class distribution
# cv=5 = 5 folds
# scoring='f1_macro' = macro F1 (treats all classes equally)
#
# This gives a much more honest picture of model performance
# ---------------------------------------------------------------
print("\n" + "=" * 55)
print("STEP 5: Stratified 5-Fold Cross Validation")
print("=" * 55)
print("Running 5 folds... (may take 30-60 seconds)")

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

print(f"F1 Score per fold : {[round(s, 4) for s in cv_scores]}")
print(f"Mean F1 Score     : {cv_scores.mean():.4f}")
print(f"Std Deviation     : {cv_scores.std():.4f}")
print("(Low std = model is consistent across different data splits)")


# ---------------------------------------------------------------
# STEP 6: Final Evaluation on Test Set
#
# Now we use the model on X_test_pca — data it has NEVER seen.
# This is the honest final score.
#
# classification_report gives precision, recall, F1 per class
# confusion_matrix shows which classes were confused with which
# ---------------------------------------------------------------
print("\n" + "=" * 55)
print("STEP 6: Final Evaluation on Test Set")
print("=" * 55)

y_pred = model.predict(X_test_pca)

# Overall accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc*100:.2f}%")

# Detailed per-class report
print("\nDetailed Classification Report:")
print("-" * 55)
print(classification_report(
    y_test,
    y_pred,
    target_names=encoder.classes_
))

# Explanation of report columns:
print("Column meanings:")
print("  precision : when model predicts X, how often is it correct?")
print("  recall    : of all actual X cases, how many did model find?")
print("  f1-score  : harmonic mean of precision & recall (best single metric)")
print("  support   : how many samples of this class in test set")


# ---------------------------------------------------------------
# STEP 7: Confusion Matrix
#
# A confusion matrix shows:
#   Rows    = actual class
#   Columns = predicted class
#   Diagonal = correct predictions (we want these to be HIGH)
#   Off-diagonal = mistakes (we want these to be LOW)
#
# Example reading:
#   Row "glioblastoma", Col "ependymoma" = 2
#   means: 2 glioblastoma samples were wrongly predicted as ependymoma
# ---------------------------------------------------------------
print("\n" + "=" * 55)
print("STEP 7: Confusion Matrix")
print("=" * 55)

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
plt.show()
print("Chart saved: outputs/confusion_matrix.png")


# ---------------------------------------------------------------
# STEP 8: Save the trained model and PCA
#
# We save:
#   model.pkl → the trained Random Forest
#   pca.pkl   → the fitted PCA transformer
#
# These are needed in:
#   - app.py for predictions on new data
#   - biomarker script for feature importance analysis
# ---------------------------------------------------------------
print("\n" + "=" * 55)
print("STEP 8: Saving Model and PCA")
print("=" * 55)

pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(pca,   open("model/pca.pkl",   "wb"))

print("Saved: model/model.pkl")
print("Saved: model/pca.pkl")

print("\n" + "=" * 55)
print("✅ TRAINING COMPLETE — SUMMARY")
print("=" * 55)
print(f"  Features used       : {N_COMPONENTS} PCA components (from 54,613 genes)")
print(f"  Training samples    : {X_train_pca.shape[0]}")
print(f"  Test samples        : {X_test_pca.shape[0]}")
print(f"  Cross-val F1 (mean) : {cv_scores.mean():.4f}")
print(f"  Test Accuracy       : {acc*100:.2f}%")
print("\n👉 Next: Run 4_biomarker_discovery.py")